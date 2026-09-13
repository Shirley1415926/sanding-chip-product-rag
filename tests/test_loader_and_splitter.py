from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sanding_rag.markdown_loader import MetadataValidationError, load_markdown
from sanding_rag.splitter import SemanticRecursiveSplitter


class MarkdownLoaderAndSplitterTests(unittest.TestCase):
    def test_required_metadata_is_preserved_on_each_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "product.md"
            path.write_text(
                """---
source: sample/product.md
source_url: https://txs.wyfdev.com/product/test/
product_id: marketplace-test
document_type: product
updated_at: 2026-09-13
---
# 产品

## 规格
"""
                + "参数说明。" * 100,
                encoding="utf-8",
            )
            chunks = SemanticRecursiveSplitter(chunk_size=120, overlap=20).split(load_markdown(path))
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "sample/product.md")
            self.assertEqual(chunk.metadata["source_url"], "https://txs.wyfdev.com/product/test/")
            self.assertEqual(chunk.metadata["product_id"], "marketplace-test")
            self.assertEqual(chunk.metadata["document_type"], "product")
            self.assertEqual(chunk.metadata["updated_at"], "2026-09-13")

    def test_missing_required_metadata_fails_before_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "broken.md"
            path.write_text("---\nsource: x.md\nsource_url: https://txs.wyfdev.com/product/test/\n---\n# 文本", encoding="utf-8")
            with self.assertRaises(MetadataValidationError):
                load_markdown(path)


if __name__ == "__main__":
    unittest.main()
