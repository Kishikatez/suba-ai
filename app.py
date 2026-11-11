
import json
import os
import re

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLACEHOLDER_VALUES = {"", "-", "n/a", "na", "none", "null"}
RELEVANCE_THRESHOLD = 0.15


# ----------------------------------
# 1. LOAD LITERATURE
# ----------------------------------

def load_kurals():
    path = os.path.join(DATA_DIR, "thirukkural.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("kural", []) if isinstance(data, dict) else []


def load_chanakya():
    path = os.path.join(DATA_DIR, "chanakyanidhi.json")

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return data.get("verses", []) if isinstance(data, dict) else []


kurals = load_kurals()
chanakya = load_chanakya()


def meaningful_text(value):
    return isinstance(value, str) and value.strip().lower() not in PLACEHOLDER_VALUES


def normalize_kural(item):
    if not isinstance(item, dict):
        return None

    try:
        number = int(item["Number"])
    except (KeyError, TypeError, ValueError):
        return None

    line1 = item.get("Line1", "")
    line2 = item.get("Line2", "")
    if not meaningful_text(line1) and not meaningful_text(line2):
        return None

    explanation = next(
        (item.get(key, "") for key in ("mv", "sp", "mk", "explanation") if meaningful_text(item.get(key, ""))),
        ""
    )
    search_parts = [line1, line2, explanation, item.get("Translation", "")]
    chapter = item.get("Chapter", "")

    return {
        "collection": "திருக்குறள்",
        "number": number,
        "chapter": chapter.strip() if meaningful_text(chapter) else "",
        "chapter_number": (number - 1) // 10 + 1,
        "line1": line1.strip() if meaningful_text(line1) else "",
        "line2": line2.strip() if meaningful_text(line2) else "",
        "text": "\n".join(part for part in (line1, line2) if meaningful_text(part)),
        "explanation": explanation,
        "translation": item.get("Translation", "") if meaningful_text(item.get("Translation", "")) else "",
        "search_text": " ".join(part for part in search_parts if meaningful_text(part))
    }


def normalize_chanakya(item, fallback_number):
    if not isinstance(item, dict):
        return None

    verse_text = next(
        (item.get(key, "") for key in ("text", "verse", "Tamil") if meaningful_text(item.get(key, ""))),
        ""
    )
    if not meaningful_text(verse_text):
        return None

    verse_id = item.get("verse_id", fallback_number)
    chapter = item.get("chapter", "")
    meaning = item.get("meaning", "")
    source = item.get("source", "")

    return {
        "collection": "சாணக்கியநீதி",
        "number": str(verse_id),
        "chapter": chapter.strip() if meaningful_text(chapter) else "",
        "chapter_number": "",
        "line1": verse_text.strip(),
        "line2": "",
        "text": verse_text.strip(),
        "explanation": meaning.strip() if meaningful_text(meaning) else "",
        "translation": "",
        "source": source.strip() if meaningful_text(source) else "",
        "search_text": " ".join(part.strip() for part in (verse_text, meaning) if meaningful_text(part))
    }


# ----------------------------------
# 2. PREPARE THIRUKKURAL RECORDS
# ----------------------------------

kurals = [record for item in kurals if (record := normalize_kural(item))]


# ----------------------------------
# 3. PREPARE CHANAKYANIDHI RECORDS
# ----------------------------------

chanakya = [
    record for index, item in enumerate(chanakya, start=1)
    if (record := normalize_chanakya(item, index))
]

records = kurals + chanakya


# ----------------------------------
# 4. BUILD LIGHTWEIGHT SEARCH ENGINE
# ----------------------------------

def build_index(literature):
    if not literature:
        return None, None

    index_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=60000,
        sublinear_tf=True
    )

    return index_vectorizer, index_vectorizer.fit_transform(
        [record["search_text"] for record in literature]
    )


kural_vectorizer, kural_matrix = build_index(kurals)
chanakya_vectorizer, chanakya_matrix = build_index(chanakya)


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
    if not isinstance(query, str) or not query.strip():
        return []

    # Exact Kural lookup
    number = exact_kural_number(query)

    if number is not None:
        for record in kurals:
            if record["number"] == number:
                return [format_result(record, 1.0)]

        return []

    expanded = expand_query(query)
    results = []

    for literature, index_vectorizer, index_matrix in (
        (kurals, kural_vectorizer, kural_matrix),
        (chanakya, chanakya_vectorizer, chanakya_matrix)
    ):
        if not literature or index_vectorizer is None or index_matrix is None:
            continue

        query_vector = index_vectorizer.transform([expanded])

        scores = cosine_similarity(query_vector, index_matrix).flatten()
        best_indices = scores.argsort()[::-1]

        for index in best_indices:
            score = float(scores[index])
            if score < RELEVANCE_THRESHOLD:
                break
            results.append(format_result(literature[index], score))

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:limit]


def format_result(record, score):

    reference_parts = [record["collection"]]
    if record.get("chapter"):
        reference_parts.append(record["chapter"])
    elif record.get("chapter_number"):
        reference_parts.append(f'அதிகாரம் எண்: {record["chapter_number"]}')
    reference_parts.append(f'எண்: {record["number"]}')

    return {
        "collection": record["collection"],
        "reference": " | ".join(reference_parts),
        "number": record["number"],
        "chapter": record["chapter"],
        "chapter_number": record.get("chapter_number", ""),
        "line1": record.get("line1", ""),
        "line2": record.get("line2", ""),
        "text": record["text"],
        "explanation": record["explanation"],
        "translation": record["translation"],
        "source": record.get("source", ""),
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
