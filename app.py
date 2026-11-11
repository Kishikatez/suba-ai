
import json
import os
import re

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_DIR = "data"


# ----------------------------------
# 1. LOAD LITERATURE
# ----------------------------------

def load_kurals():
    path = os.path.join(DATA_DIR, "thirukkural.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("kural", [])


def load_chanakya():
    path = os.path.join(DATA_DIR, "chanakyanidhi.json")

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return data.get("verses", [])


kurals = load_kurals()
chanakya = load_chanakya()


# ----------------------------------
# 2. PREPARE THIRUKKURAL RECORDS
# ----------------------------------

records = []

for k in kurals:
    records.append({
        "collection": "திருக்குறள்",
        "number": int(k["Number"]),
        "chapter": k.get("Chapter", ""),
        "text": k.get("Line1", "") + "\n" + k.get("Line2", ""),
        "explanation": (
            k.get("mv", "")
            or k.get("sp", "")
            or k.get("mk", "")
        ),
        "translation": k.get("Translation", ""),
        "search_text": " ".join([
            k.get("Line1", ""),
            k.get("Line2", ""),
            k.get("mv", ""),
            k.get("sp", ""),
            k.get("mk", ""),
            k.get("Translation", "")
        ])
    })


# ----------------------------------
# 3. PREPARE CHANAKYANIDHI RECORDS
# ----------------------------------

for index, item in enumerate(chanakya):
    if not isinstance(item, dict):
        continue

    verse_text = (
        item.get("text")
        or item.get("verse")
        or item.get("Tamil")
        or ""
    )

    if not verse_text.strip():
        continue

    records.append({
        "collection": "சாணக்கியநீதி",
        "number": item.get("verse_id", index + 1),
        "chapter": item.get("chapter", ""),
        "text": verse_text,
        "explanation": item.get("meaning", ""),
        "translation": "",
        "search_text": verse_text + " " + item.get("meaning", "")
    })


# ----------------------------------
# 4. BUILD LIGHTWEIGHT SEARCH ENGINE
# ----------------------------------

vectorizer = None
matrix = None

if records:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=60000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        [r["search_text"] for r in records]
    )


# ----------------------------------
# 5. EXACT KURAL NUMBER DETECTION
# ----------------------------------

def exact_kural_number(query):

    patterns = [
        r"திருக்குறள்\s*(?:எண்)?\s*(\d+)",
        r"kural\s*(?:number)?\s*(\d+)",
        r"குறள்\s*(\d+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            return int(match.group(1))

    return None


# ----------------------------------
# 6. SITUATION KEYWORD EXPANSION
# ----------------------------------

SITUATIONS = {
    "friendship": [
        "நட்பு", "நண்பன்", "நண்பர்கள்",
        "நண்பர்", "உறவு", "நல்ல நண்பன்",
        "நட்பாராய்தல்"
    ],
    "betrayal": [
        "துரோகம்", "வஞ்சகம்", "பொய்",
        "கபடம்", "பகை", "நட்பு",
        "கூடாநட்பு", "தீய நட்பு"
    ],
    "patience": [
        "பொறுமை", "அமைதி", "சினம்",
        "கோபம்", "அடக்கம்"
    ],
    "success": [
        "வெற்றி", "முயற்சி", "உழைப்பு",
        "ஊக்கம்", "தன்னம்பிக்கை"
    ],
    "love": [
        "காதல்", "அன்பு", "பாசம்",
        "பிரிவு", "காதலன்"
    ],
    "knowledge": [
        "கல்வி", "அறிவு", "கற்றல்",
        "ஞானம்", "படிப்பு"
    ]
}


def expand_query(query):
    expanded = query

    for keywords in SITUATIONS.values():
        if any(word in query for word in keywords):
            expanded += " " + " ".join(keywords)

    return expanded


# ----------------------------------
# 7. SEARCH ENGINE
# ----------------------------------

def search_wisdom(query, limit=5):

    if not records or vectorizer is None:
        return []

    # Exact Kural lookup
    number = exact_kural_number(query)

    if number is not None:
        for record in records:
            if (
                record["collection"] == "திருக்குறள்"
                and record["number"] == number
            ):
                return [format_result(record, 1.0)]

        return []

    # Situation-based search
    expanded = expand_query(query)

    query_vector = vectorizer.transform([expanded])

    scores = cosine_similarity(
        query_vector, matrix
    ).flatten()

    best_indices = scores.argsort()[::-1]

    results = []

    for index in best_indices:

        score = float(scores[index])

        # Reject weak matches
        if score < 0.08:
            continue

        record = records[index]

        results.append(format_result(record, score))

        if len(results) >= limit:
            break

    return results


def format_result(record, score):

    return {
        "collection": record["collection"],
        "reference": (
            f'{record["collection"]} | '
            f'எண்: {record["number"]}'
        ),
        "number": record["number"],
        "chapter": record["chapter"],
        "text": record["text"],
        "explanation": record["explanation"],
        "translation": record["translation"],
        "score": round(score, 3)
    }


# ----------------------------------
# 8. FLASK ROUTES
# ----------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        kural_count=len(kurals),
        chanakya_count=len(chanakya)
    )


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    query = data.get("message", "").strip()

    if not query:
        return jsonify({
            "message": "ஏதாவது கேளுங்கள்!",
            "results": []
        })

    results = search_wisdom(query)

    if not results:
        return jsonify({
            "message": (
                "இந்தக் கேள்விக்கு சரியான பொருத்தமான "
                "குறள் அல்லது சரிபார்க்கப்பட்ட சாணக்கியநீதி "
                "செய்யுள் கிடைக்கவில்லை. வேறு விதமாகக் கேளுங்கள்."
            ),
            "results": []
        })

    return jsonify({
        "message": "உங்கள் கேள்விக்குப் பொருத்தமான ஞானக் கருத்துகள்:",
        "results": results
    })


# ----------------------------------
# 9. START SUBA
# ----------------------------------

if __name__ == "__main__":

    print("\n========== SUBA AI ==========")
    print("Thirukkural:", len(kurals))
    print("Chanakyanidhi:", len(chanakya))
    print("Total records:", len(records))
    print("============================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
