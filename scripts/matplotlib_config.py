"""Shared matplotlib settings for repository-generated figures."""

from __future__ import annotations


def configure_matplotlib_pdf_fonts() -> None:
    """Embed editable TrueType fonts in PDF/PS outputs instead of Type 3 fonts."""
    try:
        import matplotlib
    except ImportError:
        return

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
