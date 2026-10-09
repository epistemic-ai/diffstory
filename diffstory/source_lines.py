"""Split physical source lines without interpreting Unicode text as newlines."""

import re

_EOL = re.compile(r"\r\n|\r|\n")


def split_source_lines(text: str, *, keepends: bool = False) -> list[str]:
    """Split on CRLF, CR, or LF, preserving exact endings when requested.

    Empty input has no lines. A terminal delimiter ends its preceding line;
    it does not add an empty line. All other characters remain inside lines.
    """
    lines = []
    start = 0
    for match in _EOL.finditer(text):
        lines.append(text[start : match.end() if keepends else match.start()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines
