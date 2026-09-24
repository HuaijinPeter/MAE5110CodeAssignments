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
import matplotlib
import weasyprint

matplotlib.use("Agg")

# Imported after the backend is chosen, so no display is ever needed.
import matplotlib.pyplot as plt

BLOCK_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)")

BLOCK_MATH_FONTSIZE = 13
INLINE_MATH_FONTSIZE = 11

REPORT_STYLE = """
@page { size: A4; margin: 2cm 1.8cm; }
body { font-family: "DejaVu Sans", sans-serif; font-size: 10.5pt; line-height: 1.5; }
h1 { font-size: 20pt; border-bottom: 2px solid #23699b; padding-bottom: 4px; }
h2 { font-size: 14pt; color: #23699b; margin-top: 1.4em; }
h3 { font-size: 11.5pt; margin-top: 1.2em; }
img { max-width: 100%; display: block; margin: 0.8em auto; }
img.inline-math { display: inline; margin: 0 1px; vertical-align: -0.25em; }
p.block-math { text-align: center; margin: 1.1em 0; }
p.block-math img { display: inline; }
table { border-collapse: collapse; margin: 1em auto; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 4px 9px; text-align: right; }
th { background: #eef3f7; }
td:first-child, th:first-child { text-align: left; }
code { background: #f3f3f3; padding: 1px 3px; font-size: 9.5pt; }
pre { background: #f6f6f6; padding: 8px; font-size: 9pt; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 12px; }
"""


def render_math_to_data_uri(latex, fontsize):
    """Rasterise one LaTeX snippet with mathtext and return a data URI."""
    figure = plt.figure(figsize=(0.01, 0.01))
    figure.text(0, 0, f"${latex}$", fontsize=fontsize)

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(figure)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def render_math_to_html(latex, block):
    """Return the HTML for one equation, falling back to its source."""
    fontsize = BLOCK_MATH_FONTSIZE if block else INLINE_MATH_FONTSIZE

    try:
        source = render_math_to_data_uri(latex, fontsize)
    except (ValueError, RuntimeError) as error:
        print(f"  could not render '{latex.strip()[:50]}': {error}")
        return f"<code>{latex.strip()}</code>"

    css_class = "block-math-image" if block else "inline-math"

    return f'<img class="{css_class}" src="{source}" alt="{latex.strip()}">'


def extract_math(text):
    """
    Replace every equation with a placeholder, before Markdown runs.

    Markdown would otherwise read subscripts as emphasis and mangle the
    LaTeX; the placeholders are put back once the HTML exists.
    """
    equations = []

    def take(match, block):
        equations.append((match.group(1), block))
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
