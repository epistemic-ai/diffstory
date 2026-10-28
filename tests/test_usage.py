"""Check local model-token estimates and safe request-capacity fallbacks."""

from __future__ import annotations

import io
import unittest
import uuid
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from unittest.mock import Mock
from unittest.mock import patch

import tiktoken

from diffstory.usage import INPUT_FRAMING_MARGIN
from diffstory.usage import count_input_tokens
from diffstory.usage import estimate_input


class UsageEstimationTests(unittest.TestCase):
    """Keep local input estimates bounded and safe for source preflight."""

    def _fixture_model(self, label: str) -> str:
        """Return a unique registry key to isolate the model lookup cache."""
        return f"{label}-{uuid.uuid4().hex}"

    def test_known_registry_model_uses_ordinary_text_encoding(self) -> None:
        """A locally resolved model encoding counts the exact request text."""
        text = "<|endoftext|> café 🧬"
        body = text.encode("utf-8")
        encoding = Mock()
        encoding.encode_ordinary.return_value = [0, 1, 2]

        model = self._fixture_model("gpt-4o-fixture-known")
        with patch(
            "tiktoken.encoding_for_model", return_value=encoding
        ) as resolve:
            actual = count_input_tokens(body, model=model)

        self.assertEqual(actual, 3)
        resolve.assert_called_once_with(model)
        encoding.encode_ordinary.assert_called_once_with(text)

    def test_real_tiktoken_encoding_counts_unicode_and_special_spelling(
        self,
    ) -> None:
        """The real encoder treats Unicode and special-token spelling as text."""
        text = "<|endoftext|> café 🧬"
        raw_text = text.encode("utf-8")
        mergeable_ranks = {bytes([value]): value for value in range(256)}
        for size in range(2, len(raw_text) + 1):
            mergeable_ranks[raw_text[:size]] = 254 + size
        encoding = tiktoken.Encoding(
            name="in-memory-ordinary-text-test",
            pat_str=r"(?s).+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={
                "<|endoftext|>": max(mergeable_ranks.values()) + 1
            },
        )

        with patch(
            "tiktoken.encoding_for_model", return_value=encoding
        ) as resolve:
            actual = count_input_tokens(
                raw_text, model=self._fixture_model("fixture-real-encoding")
            )

        self.assertEqual(actual, 1)
        self.assertLess(actual, len(raw_text))
        resolve.assert_called_once()

    def test_unknown_models_keep_and_cache_the_byte_estimate(self) -> None:
        """Unknown registry entries, including Terra, retain strict byte sizing."""
        terra_body = b"private source"
        with patch(
            "tiktoken.encoding_for_model", side_effect=KeyError("unmapped")
        ):
            self.assertEqual(
                count_input_tokens(terra_body, model="gpt-6.1-terra"),
                len(terra_body),
            )

        model = self._fixture_model("unknown-model-fixture")
        body = "café".encode()
        with patch(
            "tiktoken.encoding_for_model", side_effect=KeyError(model)
        ) as resolve:
            self.assertEqual(count_input_tokens(body, model=model), len(body))
            self.assertEqual(count_input_tokens(body, model=model), len(body))
            resolve.assert_called_once_with(model)

    def test_missing_vocab_fails_silently_and_caches_fallback(self) -> None:
        """Unavailable public vocabularies fall back without logging source."""
        body = b"private source in request body"
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        with (
            redirect_stdout(captured_stdout),
            redirect_stderr(captured_stderr),
            patch(
                "tiktoken.encoding_for_model",
                side_effect=OSError(body.decode()),
            ) as resolve,
        ):
            model = self._fixture_model("gpt-4o-fixture-unavailable")
            actual = count_input_tokens(body, model=model)
            repeated = count_input_tokens(body, model=model)

        self.assertEqual(actual, len(body))
        self.assertEqual(repeated, len(body))
        resolve.assert_called_once_with(model)
        self.assertEqual(captured_stdout.getvalue(), "")
        self.assertEqual(captured_stderr.getvalue(), "")

    def test_encoder_failure_uses_bytes(self) -> None:
        """A failing local encoder must not invalidate source preflight."""
        body = b"<|endoftext|> source text"
        encoding = Mock(
            encode_ordinary=Mock(side_effect=OSError("offline tokenizer"))
        )
        model = self._fixture_model("fixture-encoder-failure")
        with patch("tiktoken.encoding_for_model", return_value=encoding):
            self.assertEqual(count_input_tokens(body, model=model), len(body))

    def test_invalid_utf8_keeps_bytes_without_loading_model_data(self) -> None:
        """Binary request bodies bypass model lookup and use their byte size."""
        body = b"\xff\xfe"
        with patch("tiktoken.encoding_for_model") as resolve:
            self.assertEqual(
                count_input_tokens(
                    body,
                    model=self._fixture_model("fixture-invalid-utf8"),
                ),
                len(body),
            )
        resolve.assert_not_called()

    def test_estimate_adds_existing_framing_margins(self) -> None:
        """Model estimates preserve the fixed and adapter framing margins."""
        body = b"a serialized request"
        encoding = Mock()
        encoding.encode_ordinary.return_value = [0, 1, 2, 3]
        model = self._fixture_model("fixture-framing-estimate")
        with patch("tiktoken.encoding_for_model", return_value=encoding):
            actual = estimate_input(body, overhead_bytes=64, model=model)
        self.assertEqual(actual, 4 + INPUT_FRAMING_MARGIN + 64)

    def test_tokenizer_lookup_receives_no_request_body(self) -> None:
        """Offline lookup sees only the model ID, never source request content."""
        body = b"private source stays in this serialized request"
        model = self._fixture_model("gpt-6.1-terra-private-source-probe")
        with patch(
            "tiktoken.encoding_for_model", side_effect=KeyError("unmapped")
        ) as resolve:
            actual = estimate_input(body, model=model)
        self.assertEqual(actual, len(body) + INPUT_FRAMING_MARGIN)
        resolve.assert_called_once_with(model)


if __name__ == "__main__":
    unittest.main()
