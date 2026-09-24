# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown", "matplotlib", "pygments", "weasyprint"]
# ///
"""Compile the assignment report from Markdown to PDF.

Run with `uv run build_report.py`; uv installs the converter dependencies
into a throw-away environment, so they stay out of the project's own
dependency list. Generate the figures first with `uv run python
assignment_2.py --task <name>`.

LaTeX between dollar signs is rendered offline with Matplotlib's mathtext
and embedded as images, because WeasyPrint runs no JavaScript and so
cannot use a browser-side math renderer.
"""

import base64
import io
import re
import sys
from pathlib import Path

import markdown
import weasyprint
from matplotlib import mathtext
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")

# Mathtext's glyphs run a little smaller than the body face, so inline
# math is nudged up to match it optically rather than nominally.
INLINE_MATH_FONTSIZE = 11.0
BLOCK_MATH_FONTSIZE = 13.0

POINTS_PER_INCH = 72

REPORT_STYLE = """
@page { size: A4; margin: 2cm 1.8cm; }
body { font-family: "DejaVu Sans", sans-serif; font-size: 10.5pt; line-height: 1.5; }
h1 { font-size: 20pt; border-bottom: 2px solid #23699b; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #23699b; margin-top: 1.4em; }
h3 { font-size: 11.5pt; margin-top: 1.2em; }
img { max-width: 100%; display: block; margin: 0.8em auto; }
p.block-math { text-align: center; margin: 1.2em 0; }
table { border-collapse: collapse; margin: 1em auto; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 9px; text-align: left; }
th { background: #eef3f7; }
code { background: #f3f3f3; padding: 1px 3px; font-size: 9.5pt; }
pre { background: #f6f6f6; padding: 8px; font-size: 9pt; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 12px; }
"""


def tighten_index_spacing(latex):
    """
    Stop mathtext from spacing an index like a binary operator.

    Written plainly, `k+1` in a subscript is set with the same gaps as a
    sum; bracing the sign makes it an ordinary symbol, as TeX users do.
    """
    for loose, tight in (("_{k+1}", "_{k{+}1}"), ("_{k-1}", "_{k{-}1}")):
        latex = latex.replace(loose, tight)

    return latex


def promote_display_fractions(latex):
    """Switch fractions to display style, as LaTeX does in display math."""
    return latex.replace(r"\dfrac", r"\frac").replace(r"\frac", r"\dfrac")


def measure_math(latex, fontsize):
    """Return the width, height and baseline depth of one snippet, in points."""
    parse = mathtext.MathTextParser("path").parse(
        f"${latex}$",
        dpi=POINTS_PER_INCH,
        prop=FontProperties(size=fontsize),
    )

    return parse.width, parse.height, parse.depth


def render_math_to_data_uri(latex, fontsize, width, height, depth):
    """
    Draw one snippet as an SVG sized to exactly (width, height) points.

    The canvas is laid out in points rather than cropped to the ink, so
    the SVG carries its true typographic size and the baseline sits a
    known ``depth`` above its bottom edge. That is what lets the HTML
    place the image inline at the same scale as the surrounding text.
    """
    figure = Figure(figsize=(width / POINTS_PER_INCH, height / POINTS_PER_INCH))
    figure.patch.set_alpha(0.0)

    figure.text(
        0.0,
        depth / height,
        f"${latex}$",
        fontsize=fontsize,
        va="baseline",
        ha="left",
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", transparent=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/svg+xml;base64,{encoded}"


def render_math_to_html(latex, block):
    """Return the HTML for one equation, falling back to its source."""
    fontsize = BLOCK_MATH_FONTSIZE if block else INLINE_MATH_FONTSIZE

    latex = tighten_index_spacing(latex)

    if block:
        latex = promote_display_fractions(latex)

    try:
        width, height, depth = measure_math(latex, fontsize)
        source = render_math_to_data_uri(latex, fontsize, width, height, depth)
    except (ValueError, RuntimeError) as error:
        print(f"  could not render '{latex.strip()[:50]}': {error}")
        return f"<code>{latex.strip()}</code>"

    if block:
        # Let a wide equation shrink to the text column instead of
        # overflowing it; the height follows to keep the aspect ratio.
        style = f"width:{width:.1f}pt;max-width:100%;height:auto;display:inline"
    else:
        style = (
            f"width:{width:.1f}pt;height:{height:.1f}pt;"
            f"vertical-align:-{depth:.1f}pt;display:inline;margin:0 1px"
        )

    return f'<img src="{source}" style="{style}" alt="{latex.strip()}">'


def extract_math(text):
    """
    Replace every equation with a placeholder, before Markdown runs.

    Markdown would otherwise read subscripts as emphasis and mangle the
    LaTeX; the placeholders are put back once the HTML exists.
    """
    equations = []

    def take(match, block):
        # LaTeX ignores line breaks, but the math parser does not, so
        # equations wrapped across source lines are joined up first.
        equations.append((" ".join(match.group(1).split()), block))
        placeholder = f"MATHPLACEHOLDER{len(equations) - 1}X"

        return f"\n\n{placeholder}\n\n" if block else placeholder

    text = BLOCK_MATH.sub(lambda match: take(match, True), text)
    text = INLINE_MATH.sub(lambda match: take(match, False), text)

    return text, equations


def restore_math(html, equations):
    """Swap each placeholder for its rendered equation."""
    for index, (latex, block) in enumerate(equations):
        placeholder = f"MATHPLACEHOLDER{index}X"
        rendered = render_math_to_html(latex, block)

        if block:
            html = html.replace(
                f"<p>{placeholder}</p>",
                f'<p class="block-math">{rendered}</p>',
            )

        html = html.replace(placeholder, rendered)

    return html


def render_markdown_to_html(markdown_path):
    """Convert one Markdown file into a standalone styled HTML document."""
    source, equations = extract_math(
        markdown_path.read_text(encoding="utf-8")
    )

    body = markdown.markdown(
        source,
        extensions=["tables", "fenced_code", "codehilite", "attr_list"],
    )

    body = restore_math(body, equations)

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{REPORT_STYLE}</style></head><body>{body}</body></html>"
    )


def build_report(markdown_path, pdf_path):
    """Write the report PDF next to the project's other outputs."""
    html = render_markdown_to_html(markdown_path)

    weasyprint.HTML(
        string=html,
        base_url=str(markdown_path.parent),
    ).write_pdf(pdf_path)


def main():
    markdown_path = Path("assignment_2_report.md")
    pdf_path = Path("output/assignment_2_report.pdf")

    if not markdown_path.exists():
        sys.exit(f"{markdown_path} not found.")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    build_report(markdown_path, pdf_path)

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
