from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.llm import OpenAICompatibleLLM


class OpenAICompatibleLLMErrorTests(unittest.TestCase):
    def test_http_error_preserves_a_short_provider_diagnostic(self) -> None:
        error = HTTPError(
            url="https://api.example.test/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"model is unavailable"}}'),
        )
        llm = OpenAICompatibleLLM(
            api_base="https://api.example.test",
            api_key="test-key",
            model="test-model",
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, r"HTTP 400; model is unavailable"):
                llm.answer("question", "context", "handoff")


if __name__ == "__main__":
    unittest.main()
