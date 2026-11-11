import unittest

import app


class RetrievalTests(unittest.TestCase):
    def test_parent_voice_typo_returns_three_kurals(self):
        response = app.app.test_client().post(
            "/chat", json={"message": "பெற்றோர் மூன்று குரல் கூறுக"}
        )

        payload = response.get_json()
        self.assertEqual(len(payload["results"]), 3)
        self.assertEqual(
            [result["number"] for result in payload["results"]], [67, 69, 70]
        )
        self.assertIn("மூன்று", payload["message"])

    def test_follow_up_uses_recent_conversation_context(self):
        response = app.app.test_client().post(
            "/chat",
            json={
                "message": "அதற்கு ஏன்?",
                "conversation": [
                    {"role": "user", "text": "நல்ல நண்பனை எப்படி தேர்ந்தெடுப்பது?"}
                ]
            }
        )

        payload = response.get_json()
        self.assertTrue(payload["results"])

    def test_situation_response_is_empathetic(self):
        response = app.app.test_client().post(
            "/chat", json={"message": "எனக்கு ரொம்ப கஷ்டமா இருக்கு"}
        )

        self.assertIn("கஷ்டமாக", response.get_json()["message"])

    def test_tanglish_is_translated_before_basic_response(self):
        response = app.app.test_client().post("/chat", json={"message": "vanakkam"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["translated_query"], "வணக்கம்")

    def test_english_friend_question_uses_tamil_retrieval(self):
        response = app.app.test_client().post(
            "/chat", json={"message": "how to choose a good friend"}
        )

        payload = response.get_json()
        self.assertEqual(payload["translated_query"], "நல்ல நண்பனை எப்படி தேர்ந்தெடுப்பது")
        self.assertTrue(payload["results"])

    def test_basic_greeting_gets_conversational_response(self):
        response = app.app.test_client().post("/chat", json={"message": "வணக்கம்"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("SUBA", payload["message"])
        self.assertEqual(payload["results"], [])

    def test_thirukkural_definition_gets_basic_answer(self):
        response = app.app.test_client().post(
            "/chat", json={"message": "திருக்குறள் என்றால் என்ன?"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("திருவள்ளுவர்", payload["message"])
        self.assertIn("1330", payload["message"])
        self.assertEqual(payload["results"], [])

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

    def test_friend_selection_question_returns_kural(self):
        results = app.search_wisdom("நல்ல நண்பனை எப்படி தேர்ந்தெடுப்பது?")

        self.assertTrue(results)
        self.assertTrue(any(result["number"] in {786, 791, 792} for result in results))
        self.assertTrue(all(result["score"] >= app.RELEVANCE_THRESHOLD for result in results))

    def test_exact_chapter_lookup_returns_the_chapter(self):
        results = app.search_wisdom("திருக்குறள் அதிகாரம் 10")

        self.assertEqual(len(results), 10)
        self.assertEqual([result["number"] for result in results], list(range(91, 101)))
        self.assertTrue(all(result["chapter_number"] == 10 for result in results))

    def test_empty_chanakyanidhi_data_is_not_returned(self):
        self.assertTrue(app.chanakya)
        results = app.search_wisdom("நட்பு")

        self.assertTrue(any(result["collection"] == "சாணக்கியநீதி" for result in results))
        self.assertTrue(all(result["source"] for result in results if result["collection"] == "சாணக்கியநீதி"))

    def test_chanakya_sample_has_chapter_based_references(self):
        result = next(
            result for result in app.search_wisdom("நான்கு குலங்கள்")
            if result["collection"] == "சாணக்கியநீதி"
        )

        self.assertIn("அத்தியாயம்", result["reference"])
        self.assertIn("பகுதி", result["reference"])
        self.assertIn("sample.txt", result["source"])

    def test_unrelated_query_returns_no_weak_match(self):
        results = app.search_wisdom("கணினி ராக்கெட் வானிலை")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()