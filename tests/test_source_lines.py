"""Physical source lines retain all characters except explicit EOL removal."""

import unittest

from diffstory.source_lines import split_source_lines


class SourceLineTests(unittest.TestCase):
    """Distinguish source delimiters from Unicode and C0 text characters."""

    def test_physical_delimiters_and_terminal_behavior(self) -> None:
        """Match empty and terminal behavior for LF and CRLF."""
        for eol in ("\n", "\r\n"):
            for text, expected in (
                ("", []),
                (eol, [""]),
                (eol * 2, ["", ""]),
                ("a" + eol, ["a"]),
                ("a" + eol + "b", ["a", "b"]),
            ):
                with self.subTest(eol=eol, text=text):
                    self.assertEqual(split_source_lines(text), expected)
                    self.assertEqual(
                        "".join(split_source_lines(text, keepends=True)), text
                    )
        self.assertEqual(
            split_source_lines("a\r\nb\rc\n", keepends=True),
            ["a\r\n", "b\rc\n"],
        )

    def test_unicode_and_control_characters_remain_inside_lines(self) -> None:
        """Keep all nonphysical separators, including inside long lines."""
        text = "a\u2028b\u2029c\x85d\v e\f f\x1c\x1d\x1e\r" * 2000 + "tail"
        self.assertEqual(split_source_lines(text), [text])
        self.assertEqual(split_source_lines(text + "\n"), [text])
        self.assertEqual(
            split_source_lines(text + "\r\n", keepends=True), [text + "\r\n"]
        )

    def test_bare_and_repeated_carriage_returns_keep_git_line_bounds(
        self,
    ) -> None:
        """Bare CR must not invent lines or disappear at EOF or before CRLF."""
        cases = [
            ("one\rtwo\r", ["one\rtwo\r"]),
            ("one\r\n", ["one"]),
            ("one\r\r\n", ["one\r"]),
            ("one\rtwo\nnext\n", ["one\rtwo", "next"]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(split_source_lines(text), expected)
                self.assertEqual(
                    "".join(split_source_lines(text, keepends=True)), text
                )
