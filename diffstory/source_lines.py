"""Split source using Git line coordinates while preserving embedded controls."""

import re

_EOL = re.compile(r"\r\n|\n")


def split_source_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split on LF or CRLF, preserving exact endings when requested.

    Empty input has no lines. A terminal delimiter ends its preceding line;
    it does not add an empty line. Bare CR and other characters remain inside
    lines, matching Git patch and GitHub permalink coordinates.
    """
    lines = []
    start = 0
    for match in _EOL.finditer(text):
        lines.append(text[start : match.end() if keepends else match.start()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines
