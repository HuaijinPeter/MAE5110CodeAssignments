# /// script
# requires-python = ">=3.11"
# dependencies = ["pypandoc-binary"]
# ///
"""Compile the assignment report from Markdown to PDF, through LaTeX.

Run with `uv run build_report.py`. uv supplies pandoc; the LaTeX engine is
Tectonic, which has to be on PATH and fetches what it needs on first run:

    curl -sSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C ~/.local/bin

Generate the figures first with `uv run python assignment_2.py --task <name>`.
"""

import shutil
import sys
from pathlib import Path

import pypandoc

REPORT = Path("assignment_2_report.md")
PREAMBLE = Path("report_preamble.tex")
PDF = Path("output/assignment_2_report.pdf")

LATEX_ENGINE = "tectonic"

PANDOC_OPTIONS = [
    "--number-sections",
    "--variable=geometry:a4paper,margin=2.4cm",
    "--variable=fontsize:11pt",
    "--variable=colorlinks:true",
    "--variable=linkcolor:RoyalBlue",
    "--variable=urlcolor:RoyalBlue",
    "--syntax-highlighting=tango",
]


def require_latex_engine():
    """Stop with an explanation if the LaTeX engine is not installed."""
    if shutil.which(LATEX_ENGINE) is None:
        sys.exit(
            f"{LATEX_ENGINE} is not on PATH. "
            "See the installation line in this file's docstring."
        )


def build_report(markdown_path, pdf_path, preamble_path):
    """Typeset one Markdown report as a PDF."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pypandoc.convert_file(
        str(markdown_path),
        to="pdf",
        format="markdown",
        outputfile=str(pdf_path),
        extra_args=[
            f"--pdf-engine={LATEX_ENGINE}",
            f"--include-in-header={preamble_path}",
            # Image paths in the Markdown are relative to the repository.
            f"--resource-path={markdown_path.parent.resolve()}",
            *PANDOC_OPTIONS,
        ],
    )


def main():
    for path in (REPORT, PREAMBLE):
        if not path.exists():
            sys.exit(f"{path} not found.")

    require_latex_engine()

    build_report(REPORT, PDF, PREAMBLE)

    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
