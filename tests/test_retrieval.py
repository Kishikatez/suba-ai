import unittest

import app


class RetrievalTests(unittest.TestCase):
    def test_exact_kural_number_lookup(self):
        results = app.search_wisdom("திருக்குறள் 100 சொல்லு")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["collection"], "திருக்குறள்")
        self.assertEqual(results[0]["number"], 100)
        self.assertEqual(results[0]["chapter_number"], 10)
        self.assertEqual(results[0]["line1"], "இனிய உளவாக இன்னாத கூறல்")
        self.assertEqual(results[0]["line2"], "கனிஇருப்பக் காய்கவர்ந் தற்று.")
        self.assertIn("அதிகாரம் எண்: 10", results[0]["reference"])
        self.assertIn("இனிய உளவாக", results[0]["text"])

    def test_situation_search_returns_relevant_kural(self):
        results = app.search_wisdom("என் நண்பன் எனக்கு துரோகம் செய்துவிட்டான்")

        self.assertTrue(results)
        self.assertTrue(all(result["score"] >= app.RELEVANCE_THRESHOLD for result in results))
        self.assertTrue(any(result["collection"] == "திருக்குறள்" for result in results))

    def test_empty_chanakyanidhi_data_is_not_returned(self):
        self.assertEqual(app.chanakya, [])
        results = app.search_wisdom("நட்பு")

        self.assertTrue(all(result["collection"] != "சாணக்கியநீதி" for result in results))

    def test_unrelated_query_returns_no_weak_match(self):
        results = app.search_wisdom("கணினி ராக்கெட் வானிலை")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()