import polars as pl
import re

def normalized(column: str) -> pl.Expr:
    """Return a Polars expression that normalizes basic whitespace.

    Null values are replaced with empty strings, repeated whitespace is
    collapsed to one space, and leading or trailing whitespace is removed.
    """
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .fill_null("")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def clean_markdown(expr: pl.Expr) -> pl.Expr:
    """Remove common Reddit Markdown syntax from a text expression.

    URLs are replaced with ``[URL]``. Blockquotes, emphasis markers,
    inline-code markers, list prefixes, and link destinations are removed.

    Each delimiter pair is handled by its own pattern rather than an
    alternation such as ``(?:\\*\\*|__)(.+?)(?:\\*\\*|__)``, which would also
    match mismatched pairs like ``**text__``. (Backreferences would express
    this more compactly, but Polars' regex engine does not support them.)
    The function returns a Polars expression, so cleaning runs inside Polars
    rather than through a Python row loop.
    """
    replacements = [
        (r"https?://\S+", "[URL]"),          # URLs
        (r"(?m)^>.*$", ""),                  # blockquotes
        (r"~~(.+?)~~", "${1}"),              # strikethrough
        (r"\*\*(.+?)\*\*", "${1}"),          # bold
        (r"\*(.+?)\*", "${1}"),              # italic
        (r"__(.+?)__", "${1}"),              # bold alt
        (r"_(.+?)_", "${1}"),                # italic alt
        (r"`(.+?)`", "${1}"),                # inline code
        (r"(?m)^\s*[-*]\s+", ""),            # bullets
        (r"(?m)^\s*\d+\.\s+", ""),           # numbered lists
        (r"\[([^\]]+)\]\([^)]+\)", "${1}"),  # links -> labels
        (r"\n{3,}", "\n\n"),                 # repeated newlines
        (r"[ \t]+", " "),                    # repeated spaces
    ]

    for pattern, replacement in replacements:
        expr = expr.str.replace_all(pattern, replacement)

    return expr.str.strip_chars()
