
import json
import asyncio
import io
import os
import re

import edge_tts
import requests
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLACEHOLDER_VALUES = {"", "-", "n/a", "na", "none", "null"}
RELEVANCE_THRESHOLD = 0.15
TRANSLATION_TIMEOUT = 5


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

    if isinstance(data, list) and data:
        return data

    if isinstance(data, dict) and data.get("verses"):
        return data["verses"]

    return load_chanakya_sample()


def load_chanakya_sample():
    path = os.path.join(os.path.dirname(DATA_DIR), "sample.txt")
    if not os.path.exists(path):
        return []

    chapter_pattern = re.compile(
        r"^(?P<title>.*?(?:முதல்|இரண்டாம்|மூன்றாம்|நான்காம்|ஐந்தாம்|ஆறாம்)\s*"
        r"(?:அத்யாயம்|அத்தியாயம்).*)$"
    )
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    chapters = []
    current_title = None
    current_lines = []

    for line in lines:
        cleaned = line.replace("\u200c", "").replace("\u200d", "").strip()
        match = chapter_pattern.match(cleaned)
        if match:
            if current_title and current_lines:
                chapters.append((current_title, current_lines))
            current_title = match.group("title").strip()
            current_lines = []
        elif current_title:
            current_lines.append(cleaned)

    if current_title and current_lines:
        chapters.append((current_title, current_lines))

    records = []
    for chapter_index, (chapter_title, chapter_lines) in enumerate(chapters, start=1):
        paragraphs = re.split(r"\n\s*\n", "\n".join(chapter_lines))
        for section_index, paragraph in enumerate(paragraphs, start=1):
            text = re.sub(r"\s+", " ", paragraph).strip()
            if len(text) < 40 or "சாணக்கிய நீதி" in text and section_index == 1:
                continue
            records.append({
                "verse_id": f"அத்தியாயம் {chapter_index} - பகுதி {section_index}",
                "chapter": chapter_title,
                "text": text,
                "meaning": "",
                "source": "sample.txt (user-provided Tamil prose source)"
            })

    return records


kurals = load_kurals()
chanakya = load_chanakya()


def meaningful_text(value):
    return isinstance(value, str) and value.strip().lower() not in PLACEHOLDER_VALUES


TANGLISH_TRANSLATIONS = {
    "hi": "வணக்கம்",
    "hello": "வணக்கம்",
    "vanakkam": "வணக்கம்",
    "vanakam": "வணக்கம்",
    "thanks": "நன்றி",
    "thank you": "நன்றி",
    "what is thirukkural": "திருக்குறள் என்றால் என்ன",
    "what is tirukkural": "திருக்குறள் என்றால் என்ன",
    "who are you": "நீ யார்",
    "what can you do": "நீ என்ன செய்ய முடியும்",
    "how to choose a good friend": "நல்ல நண்பனை எப்படி தேர்ந்தெடுப்பது",
    "how do i choose a good friend": "நல்ல நண்பனை எப்படி தேர்ந்தெடுப்பது",
    "my friend betrayed me": "என் நண்பன் எனக்கு துரோகம் செய்துவிட்டான்"
}


def contains_tamil(text):
    return bool(re.search(r"[\u0B80-\u0BFF]", text))


def translate_to_tamil(query):
    cleaned = query.strip()
    if not cleaned or contains_tamil(cleaned):
        return cleaned

    normalized = re.sub(r"[!?.,]+", " ", cleaned.lower()).strip()
    local_translation = TANGLISH_TRANSLATIONS.get(normalized)
    if local_translation:
        return local_translation

    try:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": cleaned, "langpair": "en|ta"},
            timeout=TRANSLATION_TIMEOUT
        )
        response.raise_for_status()
        translated = response.json().get("responseData", {}).get("translatedText", "").strip()
        if translated and contains_tamil(translated):
            return translated
    except (requests.RequestException, ValueError, TypeError):
        pass

    return cleaned


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
        "chapter_number": item.get("chapter_number", ""),
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


def exact_chapter_number(query):
    patterns = [
        r"திருக்குறள்\s*அதிகாரம்\s*(?:எண்)?\s*(\d+)",
        r"அதிகாரம்\s*(?:எண்)?\s*(\d+)",
        r"chapter\s*(?:number)?\s*(\d+)"
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
    "friend_selection": [
        "நல்ல நண்பனை", "நல்ல நண்பர்", "நண்பனை தேர்ந்தெடுக்க",
        "நண்பரை தேர்ந்தெடுக்க", "நண்பனை தேர்வு", "நண்பரை தேர்வு",
        "நட்பாராய்தல்", "நட்பு ஆராய்தல்", "கேண்மை", "தேர்ந்து",
        "தேர்ந்தெடு", "நாடாது நட்டல்", "ஆய்ந்தாய்ந்து கொள்ளுதல்",
        "மாசற்றார் கேண்மை", "அறனறிந்து மூத்த அறிவுடையார் கேண்மை"
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

    chapter_number = exact_chapter_number(query)

    if chapter_number is not None:
        return [
            format_result(record, 1.0)
            for record in kurals
            if record["chapter_number"] == chapter_number
        ]

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
    if record["collection"] == "சாணக்கியநீதி" and str(record["number"]).startswith("அத்தியாயம்"):
        reference_parts.append(str(record["number"]))
    else:
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


def basic_conversation_response(query):
    normalized = re.sub(r"[!?.,؛،]+", " ", query.lower()).strip()

    if normalized in {"hi", "hello", "hey", "ஹாய்", "ஹலோ", "வணக்கம்", "வணக்கம"}:
        return "வணக்கம்! நான் SUBA. தமிழ் இலக்கிய ஞானம், திருக்குறள், வாழ்க்கைச் சூழ்நிலைகள் பற்றி என்னிடம் கேளுங்கள்."

    if normalized in {"vanakkam", "vanakam", "vanakkam suba"}:
        return "வணக்கம்! உங்களை சந்திப்பதில் மகிழ்ச்சி. இன்று என்ன தெரிந்துகொள்ள விரும்புகிறீர்கள்?"

    if (
        "திருக்குறள்" in normalized
        and any(phrase in normalized for phrase in ("என்றால் என்ன", "என்ன", "பற்றி சொல்லு", "விளக்கம்"))
    ) or normalized in {"what is thirukkural", "what is tirukkural", "thirukkural meaning"}:
        return (
            "திருக்குறள் என்பது திருவள்ளுவர் எழுதிய அறநூல். இதில் 1330 குறள்கள், "
            "133 அதிகாரங்களாகப் பிரிக்கப்பட்டுள்ளன. அறம், பொருள், இன்பம் ஆகிய வாழ்க்கைத் தலைப்புகளை இது பேசுகிறது."
        )

    if any(phrase in normalized for phrase in ("நீ யார்", "உன்னைப் பற்றி", "who are you", "what can you do")):
        return (
            "நான் SUBA. உங்கள் கேள்வியைப் புரிந்து கொண்டு, பொருத்தமான திருக்குறளைத் தேடி, "
            "அதன் பொருளை எளிய தமிழில் விளக்க உதவுகிறேன்."
        )

    if normalized in {"நன்றி", "ரொம்ப நன்றி", "thanks", "thank you", "nandri"}:
        return "மகிழ்ச்சி! இன்னும் ஏதாவது கேட்கலாம்."

    return None


def contextual_query(query, conversation):
    if not isinstance(conversation, list):
        return query

    recent_user_messages = [
        item.get("text", "").strip()
        for item in conversation[-6:]
        if isinstance(item, dict) and item.get("role") == "user" and item.get("text")
    ]
    if not recent_user_messages or len(query.split()) > 4:
        return query

    follow_up_words = {"ஏன்", "எப்படி", "அதற்கு", "மேலும்", "விளக்கவும்", "why", "how", "more"}
    if any(word in query.lower().split() for word in follow_up_words):
        return f"{recent_user_messages[-1]} {query}"

    return query


def response_message(query, results):
    if not results:
        if any(word in query for word in ("கஷ்ட", "துன்ப", "வருத்த", "தனிமை", "ஏமாற்ற")):
            return "உங்களுக்கு இது கஷ்டமாக இருக்கலாம்; புரிகிறது. இந்தச் சொல்லாக்கத்திற்கு சரிபார்க்கப்பட்ட குறள் பொருத்தம் கிடைக்கவில்லை. இன்னும் கொஞ்சம் விவரமாகச் சொல்லுங்கள்."
        return "உங்கள் கேள்வியைப் புரிந்துகொண்டேன். அதற்கு நேரடியான சரிபார்க்கப்பட்ட இலக்கியப் பொருத்தம் கிடைக்கவில்லை. இன்னும் கொஞ்சம் விவரமாகச் சொல்லுங்கள்."

    if any(word in query for word in ("கஷ்டம்", "துன்பம்", "வருத்தம்", "தனிமை", "ஏமாற்றம்")):
        return "உங்களுக்கு இது கஷ்டமாக இருக்கலாம்; புரிகிறது. இந்தச் சூழ்நிலைக்கு நெருக்கமான ஞானக் கருத்துகள் இவை:"

    if any(word in query for word in ("கோபம்", "சினம்", "எரிச்சல்")):
        return "கோபம் வருவது மனித இயல்பு. அதை நிதானமாகக் கையாள உதவும் கருத்துகள் இவை:"

    if any(word in query for word in ("நண்ப", "நட்பு", "துரோகம்", "வஞ்சகம்")):
        return "நட்பு பற்றிய உங்கள் கேள்வியைப் புரிந்துகொண்டேன். தொடர்புடைய குறள்கள் இவை:"

    return "உங்கள் கேள்விக்குப் பொருத்தமான ஞானக் கருத்துகள்:"


def parent_kural_response(query):
    normalized = query.replace("குரல்", "குறள்").lower()
    parent_words = ("பெற்றோர்", "தாய்", "தந்தை", "மகன்", "மகள்")
    request_words = ("மூன்று", "3", "கூறுக", "சொல்லு", "சொல்லுங்கள்", "குறள்")

    if not any(word in normalized for word in parent_words):
        return None
    if not any(word in normalized for word in request_words):
        return None

    selected_numbers = {67, 69, 70}
    selected = [record for record in kurals if record["number"] in selected_numbers]
    selected.sort(key=lambda record: record["number"])
    return [format_result(record, 1.0) for record in selected]


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
    conversation = data.get("conversation", [])

    if not query:
        return jsonify({
            "message": "ஏதாவது கேளுங்கள்!",
            "results": [],
            "translated_query": ""
        })

    tamil_query = translate_to_tamil(query)
    search_query = contextual_query(tamil_query, conversation)
    translated_query = tamil_query if tamil_query != query else ""

    parent_results = parent_kural_response(tamil_query)
    if parent_results:
        return jsonify({
            "message": "பெற்றோர் மற்றும் குழந்தைகள் பற்றிய மூன்று பொருத்தமான குறள்கள்:",
            "results": parent_results,
            "translated_query": translated_query
        })

    basic_response = basic_conversation_response(tamil_query)
    if basic_response:
        return jsonify({
            "message": basic_response,
            "results": [],
            "translated_query": translated_query
        })

    results = search_wisdom(search_query)

    if not results:
        return jsonify({
            "message": response_message(tamil_query, results),
            "results": [],
            "translated_query": translated_query
        })

    return jsonify({
        "message": response_message(tamil_query, results),
        "results": results,
        "translated_query": translated_query
    })


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text or len(text) > 2000:
        return jsonify({"error": "குரல் உரை காலியாக இருக்கக்கூடாது அல்லது மிக நீளமாக உள்ளது."}), 400

    async def create_audio():
        audio = io.BytesIO()
        communicate = edge_tts.Communicate(text, "ta-IN-PallaviNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
        audio.seek(0)
        return audio

    try:
        audio = asyncio.run(create_audio())
    except Exception:
        return jsonify({"error": "தமிழ் குரல் சேவையை அணுக முடியவில்லை."}), 502

    return send_file(audio, mimetype="audio/mpeg", download_name="suba-tamil.mp3")


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
