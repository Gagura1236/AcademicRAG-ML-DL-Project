The full bug audit report has been sent to the main agent. Here's a quick summary of what was found across all 10 files:

---

## 🐛 Bug Audit — Quick Summary

**28 bugs found total: 7 Critical · 15 Medium · 6 Low**

| Priority | Count | Top Issues |
|----------|-------|-----------|
| 🔴 Critical | 7 | DATA_DIR outside project, KeyError crashes in app.py from empty eval returns, LoRA never reloaded after training, AttributeError on None embedding model, OCR TypeError with None unk_token_id, IndexError on empty pages list |
| 🟡 Medium | 15 | Shared metadata dict mutation in search, fitz file handle leaks, partial downloads reused, multi-line abstracts not captured, random seed inside loop kills diversity, tokenizer tuple-pair API issues |
| 🟢 Low | 6 | Duplicate `import re`, progress callback not called on last batch, tarfile path traversal, recommendations filtered too aggressively |

**Most urgent fixes (do these first):**
1. **`config.py:27`** — `DATA_DIR` is set one directory level too high (`../data` instead of `./data`), breaking all file paths project-wide
2. **`evaluation.py:86 & 220`** — Returning `{}` or incomplete dicts causes `KeyError: 'diagnosis'` crashes in `app.py`
3. **`finetune.py:275`** — After LoRA training, `load_index()` is called but it only reloads FAISS — the new LoRA weights are **never actually loaded** into the live model
4. **`embedding_search.py:120,187`** — No None-guard before calling `.encode()` on the embedding model
5. **`pdf_parser.py:145`** — `unk_token_id` can be `None`, causing `TypeError` in HuggingFace `generate()`