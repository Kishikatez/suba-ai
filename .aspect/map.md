# Map

_Symbol index with signatures and conventions. Use to find types, functions, and coding patterns._

## Symbol Index

_Functions, classes, and exports with call relationships._

### `app.py`

| Symbol | Kind | Signature | Used In (files) |
|--------|------|-----------|----------------|
| `load_kurals` | function | `def load_kurals()` | — |
| `load_chanakya` | function | `def load_chanakya()` | — |
| `meaningful_text` | function | `def meaningful_text(value)` | — |
| `normalize_kural` | function | `def normalize_kural(item)` | — |
| `normalize_chanakya` | function | `def normalize_chanakya(item, fallback_number)` | — |
| `build_index` | function | `def build_index(literature)` | — |
| `exact_kural_number` | function | `def exact_kural_number(query)` | — |
| `expand_query` | function | `def expand_query(query)` | — |
| `search_wisdom` | function | `def search_wisdom(query, limit)` | — |
| `format_result` | function | `def format_result(record, score)` | — |
| `home` | function | `def home()` | — |
| `chat` | function | `def chat()` | — |

### `tests/test_retrieval.py`

| Symbol | Kind | Signature | Used In (files) |
|--------|------|-----------|----------------|
| `RetrievalTests` | class | `class RetrievalTests(unittest.TestCase)` | — |
| `test_exact_kural_number_lookup` | method | `def test_exact_kural_number_lookup()` | — |
| `test_situation_search_returns_relevant_kural` | method | `def test_situation_search_returns_relevant_kural()` | — |
| `test_empty_chanakyanidhi_data_is_not_returned` | method | `def test_empty_chanakyanidhi_data_is_not_returned()` | — |
| `test_unrelated_query_returns_no_weak_match` | method | `def test_unrelated_query_returns_no_weak_match()` | — |

---

## Conventions

_Naming patterns and styles. Follow these for consistency._


_Generated: 2026-11-11T07:08:30.200Z_
