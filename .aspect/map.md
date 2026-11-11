# Map

_Symbol index with signatures and conventions. Use to find types, functions, and coding patterns._

## Symbol Index

_Functions, classes, and exports with call relationships._

### `app.py`

| Symbol | Kind | Signature | Used In (files) |
|--------|------|-----------|----------------|
| `load_kurals` | function | `def load_kurals()` | — |
| `load_chanakya` | function | `def load_chanakya()` | — |
| `load_chanakya_sample` | function | `def load_chanakya_sample()` | — |
| `meaningful_text` | function | `def meaningful_text(value)` | — |
| `contains_tamil` | function | `def contains_tamil(text)` | — |
| `translate_to_tamil` | function | `def translate_to_tamil(query)` | — |
| `normalize_kural` | function | `def normalize_kural(item)` | — |
| `normalize_chanakya` | function | `def normalize_chanakya(item, fallback_number)` | — |
| `build_index` | function | `def build_index(literature)` | — |
| `exact_kural_number` | function | `def exact_kural_number(query)` | — |
| `exact_chapter_number` | function | `def exact_chapter_number(query)` | — |
| `expand_query` | function | `def expand_query(query)` | — |

_+10 more symbols_

### `tests/test_retrieval.py`

| Symbol | Kind | Signature | Used In (files) |
|--------|------|-----------|----------------|
| `RetrievalTests` | class | `class RetrievalTests(unittest.TestCase)` | — |
| `test_parent_voice_typo_returns_three_kurals` | method | `def test_parent_voice_typo_returns_three_kurals()` | — |
| `test_follow_up_uses_recent_conversation_context` | method | `def test_follow_up_uses_recent_conversation_context()` | — |
| `test_situation_response_is_empathetic` | method | `def test_situation_response_is_empathetic()` | — |
| `test_tanglish_is_translated_before_basic_response` | method | `def test_tanglish_is_translated_before_basic_response()` | — |
| `test_english_friend_question_uses_tamil_retrieval` | method | `def test_english_friend_question_uses_tamil_retrieval()` | — |
| `test_basic_greeting_gets_conversational_response` | method | `def test_basic_greeting_gets_conversational_response()` | — |
| `test_thirukkural_definition_gets_basic_answer` | method | `def test_thirukkural_definition_gets_basic_answer()` | — |
| `test_exact_kural_number_lookup` | method | `def test_exact_kural_number_lookup()` | — |
| `test_situation_search_returns_relevant_kural` | method | `def test_situation_search_returns_relevant_kural()` | — |
| `test_friend_selection_question_returns_kural` | method | `def test_friend_selection_question_returns_kural()` | — |
| `test_exact_chapter_lookup_returns_the_chapter` | method | `def test_exact_chapter_lookup_returns_the_chapter()` | — |

_+3 more symbols_

---

## Conventions

_Naming patterns and styles. Follow these for consistency._

### Function Naming

- `create_*` → `create_audio` (1 occurrences)


_Generated: 2026-11-11T08:32:49.802Z_
