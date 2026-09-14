"""
VOICE-TO-TEXT AGENT - Complete Single File
Deepgram Nova-2 + Roman Urdu (hi-Latn)
Run: streamlit run app.py
"""

import io
import os
import re
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from collections import Counter
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Voice Agent", page_icon="🎤", layout="centered")
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    DEEPGRAM_URL      = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_MODEL    = "nova-2"
    DEEPGRAM_LANGUAGE = "hi-Latn"
    DEEPGRAM_TIMEOUT  = 60

    KEYTERMS = [
        "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly",
        "NumPy", "API", "machine learning", "function",
        "variable", "class", "list", "dictionary",
        "Assalam", "Alaikum", "Namaz", "Salam", "Allah", "Quran",
        "Ramadan", "InshaAllah", "MashaAllah", "Alhamdulillah",
        "Shukriya", "Khuda Hafiz", "Allah Hafiz",
    ]

    # Number words -> digits (Roman Urdu) - EXPANDED
    NUMBER_WORDS = {
        # 1-10
        "aik": "1", "ek": "1", "ik": "1",
        "do": "2", "dou": "2",
        "teen": "3", "tin": "3",
        "chaar": "4", "char": "4",
        "paanch": "5", "panch": "5", "panc": "5",
        "chhe": "6", "chay": "6", "che": "6",
        "saat": "7", "sath": "7",
        "aath": "8", "ath": "8",
        "nau": "9", "no": "9",
        "das": "10", "dus": "10",
        # 11-19
        "gyarah": "11", "barah": "12", "terah": "13",
        "chaudah": "14", "pandrah": "15", "solah": "16",
        "satrah": "17", "atharah": "18", "unnis": "19",
        # 20s
        "bees": "20", "bis": "20",
        "ikkees": "21", "baees": "22", "teis": "23",
        "chaubees": "24", "pachees": "25", "chhabbees": "26",
        "sattaees": "27", "atthaees": "28", "untis": "29",
        # 30s
        "tees": "30", "tis": "30",
        "ikattis": "31", "battis": "32", "taintis": "33",
        "chauntis": "34", "paintees": "35", "chhattis": "36",
        "saintees": "37", "adtees": "38", "untalis": "39",
        # 40s
        "chalees": "40", "chalis": "40",
        "iktaalis": "41", "bayalis": "42", "taintalis": "43",
        "chawalis": "44", "paintaalis": "45", "chhiyalis": "46",
        "saintaalis": "47", "adtaalis": "48", "unchaas": "49",
        # 50s
        "pachaas": "50", "pachas": "50",
        "ikyawan": "51", "bawan": "52", "tirpan": "53",
        "chawwan": "54", "pachpan": "55", "chhappan": "56",
        "sattawan": "57", "atthawan": "58", "unsath": "59",
        # 60s
        "saath": "60", "sath": "60",
        "iksath": "61", "basath": "62", "tirsath": "63",
        "chausath": "64", "painsath": "65", "chhiyasath": "66",
        "sarsath": "67", "arsath": "68", "unhattar": "69",
        # 70s
        "sattar": "70", "satar": "70",
        "ikhattar": "71", "bahattar": "72", "tihattar": "73",
        "chauhattar": "74", "pachattar": "75", "chhihattar": "76",
        "sathattar": "77", "athhattar": "78", "unasi": "79",
        # 80s
        "assi": "80", "asi": "80",
        "ikyasi": "81", "bayasi": "82", "tirasi": "83",
        "chaurasi": "84", "pachasi": "85", "chhiyasi": "86",
        "sattasi": "87", "athasi": "88", "navasi": "89",
        # 90s
        "navve": "90", "nawwe": "90",
        "ikyanve": "91", "banve": "92", "tiranve": "93",
        "churanve": "94", "pachanve": "95", "chhiyanve": "96",
        "sattanve": "97", "atthanve": "98", "ninyanve": "99",
        # Multipliers
        "sau": "100", "so": "100",
        "hazaar": "1000", "hazar": "1000", "hzar": "1000",
        "lakh": "100000", "lac": "100000",
        "crore": "10000000", "karor": "10000000",
    }

    # Audio processing
    MIN_RMS_ENERGY       = 30.0
    MIN_DURATION_SECONDS = 0.5
    MAX_DURATION_SECONDS = 120
    TARGET_PEAK_DB       = -3.0
    LIMITER_THRESHOLD    = 0.95
    SOFT_BOOST_DB        = 6.0
    VAD_FRAME_MS         = 30
    MIN_SPEECH_SECONDS   = 0.2
    HIGHPASS_HZ          = 80


# ============================================================
# API KEY LOADER
# ============================================================
def get_api_key():
    key = os.getenv("DEEPGRAM_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("DEEPGRAM_API_KEY")
    except Exception:
        return None


# ============================================================
# FEATURE 1: AGC + PEAK LIMITER
# ============================================================
def apply_agc_peak_limiter(audio, sr, target_db=None, threshold=None):
    target_db = target_db if target_db is not None else Config.TARGET_PEAK_DB
    threshold = threshold if threshold is not None else Config.LIMITER_THRESHOLD
    try:
        audio = audio.astype(np.float64)
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        audio_norm = audio / peak
        rms = np.sqrt(np.mean(audio_norm ** 2) + 1e-10)
        current_db = 20 * np.log10(rms)
        gain_db = np.clip(target_db - current_db, -20, 20)
        gain = 10 ** (gain_db / 20)
        audio_agc = audio_norm * gain
        audio_limited = np.tanh(audio_agc / threshold) * threshold
        return (audio_limited * 32767).astype(np.float64)
    except Exception:
        return audio


# ============================================================
# FEATURE 2: SMART VAD + BOOST
# ============================================================
def smart_vad_boost(audio, sr, boost_db=None, frame_ms=None):
    boost_db = boost_db if boost_db is not None else Config.SOFT_BOOST_DB
    frame_ms = frame_ms if frame_ms is not None else Config.VAD_FRAME_MS
    try:
        audio = audio.astype(np.float64)
        frame_len = int(sr * frame_ms / 1000)
        if frame_len < 1:
            return audio
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return audio
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])
        noise_floor = np.percentile(energies, 10)
        threshold = max(noise_floor * 2.5, Config.MIN_RMS_ENERGY / 32767.0)
        speech_mask = energies > threshold
        speech_ratio = np.mean(speech_mask)
        overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 1.5) or speech_ratio < 0.3
        if is_soft and speech_ratio > 0.05:
            boost = 10 ** (boost_db / 20)
            audio_out = audio.copy()
            for i in range(n_frames):
                if speech_mask[i]:
                    audio_out[i*frame_len:(i+1)*frame_len] *= boost
            return audio_out
        return audio
    except Exception:
        return audio


# ============================================================
# HELPERS
# ============================================================
def highpass_filter(audio, sr, cutoff=None):
    cutoff = cutoff if cutoff is not None else Config.HIGHPASS_HZ
    try:
        nyq = 0.5 * sr
        if cutoff / nyq >= 1.0:
            return audio
        b, a = signal.butter(3, cutoff / nyq, btype='high')
        return signal.filtfilt(b, a, audio.astype(np.float64))
    except Exception:
        return audio


def normalize_audio(audio, target_peak=0.95):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak * target_peak * 32767


def is_silent(audio, rms_threshold=1.0):
    if audio is None or len(audio) == 0:
        return True
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10)
    return rms < rms_threshold


def count_speakers_estimate(audio, sr):
    try:
        frame_len = int(sr * 0.1)
        n_frames = len(audio) // frame_len
        if n_frames < 5:
            return 1
        energies = np.array([
            np.sqrt(np.mean(audio.astype(np.float64)[i*frame_len:(i+1)*frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])
        q1, q2, q3 = np.percentile(energies, [25, 50, 75])
        buckets = np.digitize(energies, [q1, q2, q3])
        transitions = np.sum(np.diff(buckets) != 0)
        estimate = min(max(1, transitions // 3 + 1), 5)
        return int(estimate)
    except Exception:
        return 1


# ============================================================
# MAIN PIPELINE
# ============================================================
def preprocess_audio(audio_bytes, apply_agc=True, apply_vad=True):
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if sr <= 0:
            return None
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        duration = len(audio) / sr
        if duration < Config.MIN_DURATION_SECONDS:
            return None
        if duration > Config.MAX_DURATION_SECONDS:
            audio = audio[:int(Config.MAX_DURATION_SECONDS * sr)]
            duration = len(audio) / sr
        if is_silent(audio):
            return None
        overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        was_soft = overall_rms < (Config.MIN_RMS_ENERGY * 1.5)
        peak = np.max(np.abs(audio))
        was_loud = peak > 25000
        if apply_agc:
            audio = apply_agc_peak_limiter(audio, sr)
        if apply_vad:
            audio = smart_vad_boost(audio, sr)
        audio = highpass_filter(audio, sr)
        audio = normalize_audio(audio)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        wav.write(buf, sr, audio)
        buf.seek(0)
        return {
            "bytes": buf.read(),
            "duration": duration,
            "sample_rate": int(sr),
            "was_soft": was_soft,
            "was_loud": was_loud,
        }
    except Exception:
        return None


# ============================================================
# FEATURE 3: SPEAKER ISOLATION
# ============================================================
def pick_dominant_speaker(deepgram_data):
    try:
        results = deepgram_data.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            return "", 0.0, 0
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return "", 0.0, 0
        words = alternatives[0].get("words", [])
        if not words:
            text = alternatives[0].get("transcript", "").strip()
            conf = float(alternatives[0].get("confidence", 0.0) or 0.0)
            return text, conf, 1
        speakers = [w.get("speaker", 0) for w in words]
        speaker_counts = Counter(speakers)
        speaker_count = len(speaker_counts)
        dominant = speaker_counts.most_common(1)[0][0]
        dominant_words = [w for w in words if w.get("speaker", 0) == dominant]
        text = " ".join(w["word"] for w in dominant_words).strip()
        conf = float(np.mean([w.get("confidence", 0.0) for w in dominant_words]))
        return text, conf, speaker_count
    except Exception:
        try:
            return deepgram_data["results"]["channels"][0]["alternatives"][0]["transcript"], 0.0, 0
        except Exception:
            return "", 0.0, 0


# ============================================================
# TEXT CONVERSION HELPERS
# ============================================================
def to_roman_urdu(text):
    """Convert Urdu script to Roman if needed."""
    if not text:
        return text
    if re.search(r'[\u0600-\u06FF]', text):
        return unidecode(text)
    return text


def convert_numbers_to_digits(text):
    """Convert Roman Urdu number words to digits."""
    if not text:
        return text

    def replace_word(match):
        word = match.group(0)
        lower = word.lower()
        if lower in Config.NUMBER_WORDS:
            return Config.NUMBER_WORDS[lower]
        return word

    return re.sub(r'\b\w+\b', replace_word, text)


def combine_number_multipliers(text):
    """
    Combine digit + multiplier into single number.
        "2 1000"   -> "2000"
        "3 100"    -> "300"
        "5 100000" -> "500000"
        "2 10000000" -> "20000000"
    """
    if not text:
        return text

    # Multipliers sorted longest-first to avoid partial match
    multipliers = [10000000, 100000, 1000, 100]

    def replace_match(match):
        num = int(match.group(1))
        mult = int(match.group(2))
        return str(num * mult)

    # Regex: digit space digit (multiplier)
    # Run repeatedly for cases like "2 3 1000" (rare)
    for _ in range(3):
        new_text = re.sub(
            r'\b(\d+)\s+(' + '|'.join(str(m) for m in multipliers) + r')\b',
            replace_match,
            text
        )
        if new_text == text:
            break
        text = new_text

    return text


# ============================================================
# DEEPGRAM TRANSCRIBE
# ============================================================
def transcribe_deepgram(audio_bytes, language=None, isolate_speaker=True):
    api_key = get_api_key()
    if not api_key:
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": "DEEPGRAM_API_KEY missing"}

    lang = language or Config.DEEPGRAM_LANGUAGE

    params = [
        ("model", Config.DEEPGRAM_MODEL),
        ("language", lang),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("numerals", "true"),
        ("diarize", "true" if isolate_speaker else "false"),
        ("utterances", "true"),
    ]

    if Config.DEEPGRAM_MODEL.startswith("nova-3"):
        keyword_param = "keyterm"
    else:
        keyword_param = "keywords"

    for term in Config.KEYTERMS:
        params.append((keyword_param, term))

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    try:
        r = requests.post(Config.DEEPGRAM_URL, params=params, headers=headers,
                          data=audio_bytes, timeout=Config.DEEPGRAM_TIMEOUT)
    except requests.RequestException as e:
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": f"Network: {e}"}

    if r.status_code != 200:
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": f"Deepgram {r.status_code}: {r.text[:300]}"}

    data = r.json()

    if isolate_speaker:
        text, conf, spk_count = pick_dominant_speaker(data)
    else:
        try:
            alt = data["results"]["channels"][0]["alternatives"][0]
            text = alt.get("transcript", "").strip()
            conf = float(alt.get("confidence", 0.0) or 0.0)
            spk_count = 1
        except Exception:
            text, conf, spk_count = "", 0.0, 0

    # Post-process text
    text = to_roman_urdu(text)
    text = convert_numbers_to_digits(text)
    text = combine_number_multipliers(text)

    if not text or len(text.strip()) < 2:
        return {
            "text": "",
            "confidence": 0.0,
            "speaker_count": spk_count,
            "success": False,
            "error": "No clear speech detected. Please speak again.",
        }

    return {
        "text": text,
        "confidence": conf,
        "speaker_count": spk_count,
        "success": True,
        "error": None,
    }


# ============================================================
# MAIN ENTRY
# ============================================================
def process_voice_input(audio_bytes):
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0,
                "speaker_count": 0,
                "error": "Audio too quiet or too short. Please speak clearly."}

    if cleaned["was_loud"]:
        print("[INFO] Loud voice detected - AGC applied")
    if cleaned["was_soft"]:
        print("[INFO] Soft voice detected - Boost applied")

    result = transcribe_deepgram(cleaned["bytes"], isolate_speaker=True)

    if not result["success"]:
        err = result.get("error") or "No clear speech detected. Please try again."
        return {"success": False, "text": "", "confidence": 0.0,
                "speaker_count": 0, "error": err}

    if result["speaker_count"] > 1:
        print(f"[INFO] {result['speaker_count']} speakers detected - Dominant picked")

    return {
        "success": True,
        "text": result["text"],
        "confidence": result["confidence"],
        "speaker_count": result["speaker_count"],
        "error": None,
    }


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎤 Voice Agent")
st.caption("Record your voice - transcription will appear below")

if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_conf" not in st.session_state:
    st.session_state.last_conf = 0.0
if "last_speakers" not in st.session_state:
    st.session_state.last_speakers = 0

audio_output = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic",
)

if audio_output and audio_output.get("bytes"):
    with st.spinner("Processing audio..."):
        result = process_voice_input(audio_output["bytes"])

    if result["success"]:
        st.session_state.last_text = result["text"]
        st.session_state.last_conf = result["confidence"]
        st.session_state.last_speakers = result["speaker_count"]
        st.success("Transcription complete!")
    else:
        st.error(f"Failed: {result['error']}")

st.divider()
st.subheader("📝 Transcription")

if st.session_state.last_text:
    st.markdown(
        f"""
        <div style="background:#1e1e2e; padding:20px; border-radius:10px;
                    color:#cdd6f4; font-size:18px; line-height:1.6;">
            {st.session_state.last_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Confidence: {st.session_state.last_conf:.2f} | "
        f"Speakers detected: {st.session_state.last_speakers}"
    )
else:
    st.info("Record something to see transcription here.")

if st.button("🗑️ Clear", use_container_width=True):
    st.session_state.last_text = ""
    st.session_state.last_conf = 0.0
    st.session_state.last_speakers = 0
    st.rerun()
