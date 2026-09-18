import io
import os
import re
import json
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from collections import Counter
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode
from pathlib import Path
from datetime import datetime

# Optional: only for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Voice Agent", page_icon="🎤", layout="centered")

# ============================================================
# CONFIG
# ============================================================
class Config:
    DEEPGRAM_URL      = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_MODEL    = "nova-2"
    DEEPGRAM_LANGUAGE = "hi-Latn"
    DEEPGRAM_TIMEOUT  = 60

    KEYTERMS = [
        "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly",
        "NumPy", "API", "machine learning", "function", "variable",
        "Assalam", "Alaikum", "Namaz", "Salam", "Allah", "Quran",
        "Ramadan", "InshaAllah", "MashaAllah", "Alhamdulillah",
        "Shukriya", "Khuda Hafiz", "Allah Hafiz", "hello", "hello 123",
        "assalamoalaikum"
    ]

    NUMBER_WORDS = {
        "aik": "1", "ek": "1", "ik": "1", "do": "2", "dou": "2",
        "teen": "3", "tin": "3", "chaar": "4", "char": "4",
        "paanch": "5", "panch": "5", "chhe": "6", "chay": "6",
        "saat": "7", "aath": "8", "nau": "9", "das": "10",
        "bees": "20", "tees": "30", "chalees": "40", "pachaas": "50",
        "saath": "60", "sattar": "70", "assi": "80", "navve": "90",
        "sau": "100", "hazaar": "1000", "lakh": "100000", "crore": "10000000",
    }

    # Soft voice & SNR
    SNR_SAFE_MIN         = 28.0
    SNR_DEGRADED_MIN     = 16.0
    MIN_RMS_ENERGY       = 22.0
    SOFT_BOOST_DB        = 11.0
    VAD_FRAME_MS         = 25
    HIGHPASS_HZ          = 80
    MIN_DURATION_SECONDS = 0.4
    MAX_DURATION_SECONDS = 120

    # Paths
    PROFILE_DIR = Path("voice_profiles")
    PROFILE_META = PROFILE_DIR / "profile_meta.json"

# ============================================================
# API KEY (Streamlit Cloud + Local compatible)
# ============================================================
def get_api_key():
    # 1. Streamlit Secrets (for Streamlit Cloud)
    try:
        key = st.secrets.get("DEEPGRAM_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 2. Environment variable (for local)
    key = os.getenv("DEEPGRAM_API_KEY")
    if key:
        return key

    return None

# ============================================================
# AUDIO UTILS
# ============================================================
def apply_agc_peak_limiter(audio, sr, target_db=-3.0, threshold=0.95):
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
        return (np.tanh(audio_agc / threshold) * threshold * 32767).astype(np.float64)
    except Exception:
        return audio

def smart_vad_boost(audio, sr, boost_db=None, frame_ms=None):
    boost_db = boost_db or Config.SOFT_BOOST_DB
    frame_ms = frame_ms or Config.VAD_FRAME_MS
    try:
        audio = audio.astype(np.float64)
        frame_len = int(sr * frame_ms / 1000)
        if frame_len < 1 or len(audio) < frame_len * 2:
            return audio
        n_frames = len(audio) // frame_len
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])
        noise_floor = np.percentile(energies, 12)
        threshold = max(noise_floor * 1.55, Config.MIN_RMS_ENERGY / 32767.0 * 0.5)
        speech_mask = energies > threshold
        speech_ratio = np.mean(speech_mask)
        overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 2.6) or speech_ratio < 0.48
        if is_soft and speech_ratio > 0.02:
            boost = 10 ** (boost_db / 20)
            audio_out = audio.copy()
            for i in range(n_frames):
                if speech_mask[i]:
                    audio_out[i*frame_len:(i+1)*frame_len] *= boost
            peak = np.max(np.abs(audio_out)) + 1e-8
            return audio_out / peak * 0.90 * 32767
        return audio
    except Exception:
        return audio

def highpass_filter(audio, sr, cutoff=80):
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
    return np.sqrt(np.mean(audio.astype(np.float64)**2) + 1e-10) < rms_threshold

def estimate_snr(audio, sr):
    try:
        frame_len = int(sr * 0.02)
        if frame_len < 1:
            return 0.0
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return 0.0
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len]**2) + 1e-10)
            for i in range(n_frames)
        ])
        noise = np.percentile(energies, 10) + 1e-10
        signal_p = np.percentile(energies, 90) + 1e-10
        return float(np.clip(20 * np.log10(signal_p / noise), 0, 80))
    except Exception:
        return 0.0

def spectral_noise_reduce(audio, sr, aggressive=False):
    try:
        f, t, Zxx = signal.stft(audio.astype(np.float64), fs=sr, nperseg=512)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        frame_energy = np.mean(mag**2, axis=0)
        noise_frames = frame_energy < np.percentile(frame_energy, 10)
        noise_profile = np.mean(mag[:, noise_frames], axis=1, keepdims=True) if np.sum(noise_frames) > 0 else np.min(mag, axis=1, keepdims=True)
        alpha = 2.3 if aggressive else 1.7
        mag_clean = np.maximum(mag - alpha * noise_profile, 0.07 * mag)
        _, audio_clean = signal.istft(mag_clean * np.exp(1j * phase), fs=sr)
        if len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
        else:
            audio_clean = audio_clean[:len(audio)]
        return audio_clean
    except Exception:
        return audio

def adaptive_preprocess(audio, sr):
    audio = audio.astype(np.float64)
    snr = estimate_snr(audio, sr)
    overall_rms = np.sqrt(np.mean(audio**2) + 1e-10)
    peak = np.max(np.abs(audio))
    is_loud = peak > 25000
    is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 2.5)
    voice_type = "loud" if is_loud else ("soft" if is_soft else "normal")
    snr_zone = "safe" if snr >= Config.SNR_SAFE_MIN else ("degraded" if snr >= Config.SNR_DEGRADED_MIN else "critical")
    diagnostics = {
        "snr_db": round(snr, 1),
        "voice_type": voice_type,
        "snr_zone": snr_zone,
        "filters_applied": []
    }
    if snr_zone == "safe":
        audio = highpass_filter(audio, sr, 80)
        diagnostics["filters_applied"].append("highpass_80")
    elif snr_zone == "degraded":
        audio = highpass_filter(audio, sr, 100)
        audio = spectral_noise_reduce(audio, sr, False)
        diagnostics["filters_applied"].append("highpass_100 + NR")
    else:
        audio = highpass_filter(audio, sr, 120)
        audio = spectral_noise_reduce(audio, sr, True)
        diagnostics["filters_applied"].append("highpass_120 + aggressive NR")
    if is_loud:
        audio = apply_agc_peak_limiter(audio, sr)
        diagnostics["filters_applied"].append("agc_limiter")
    elif is_soft:
        audio = smart_vad_boost(audio, sr)
        diagnostics["filters_applied"].append("strong_soft_boost")
    audio = normalize_audio(audio)
    return np.clip(audio, -32768, 32767).astype(np.int16), diagnostics

def preprocess_audio(audio_bytes):
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if sr <= 0:
            return None
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        duration = len(audio) / sr
        if duration < Config.MIN_DURATION_SECONDS or is_silent(audio):
            return None
        if duration > Config.MAX_DURATION_SECONDS:
            audio = audio[:int(Config.MAX_DURATION_SECONDS * sr)]
            duration = len(audio) / sr
        audio_clean, diagnostics = adaptive_preprocess(audio, sr)
        buf = io.BytesIO()
        wav.write(buf, sr, audio_clean)
        buf.seek(0)
        return {"bytes": buf.read(), "duration": duration, "diagnostics": diagnostics}
    except Exception:
        return None

# ============================================================
# SPEAKER SELECTION
# ============================================================
def pick_better_speaker(deepgram_data, last_speaker=None, locked_speaker=None):
    try:
        channels = deepgram_data.get("results", {}).get("channels", [])
        if not channels:
            return "", 0.0, 0, None
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return "", 0.0, 0, None
        words = alternatives[0].get("words", [])
        if not words:
            text = alternatives[0].get("transcript", "").strip()
            conf = float(alternatives[0].get("confidence", 0.0) or 0.0)
            return text, conf, 1, None
        speaker_data = {}
        for w in words:
            spk = w.get("speaker", 0)
            if spk not in speaker_data:
                speaker_data[spk] = {"words": [], "confidences": [], "start": w.get("start", 0), "end": w.get("end", 0)}
            speaker_data[spk]["words"].append(w.get("word", ""))
            speaker_data[spk]["confidences"].append(float(w.get("confidence", 0.0)))
            speaker_data[spk]["end"] = w.get("end", speaker_data[spk]["end"])
        scores = {}
        for spk, data in speaker_data.items():
            word_count = len(data["words"])
            avg_conf = np.mean(data["confidences"]) if data["confidences"] else 0.0
            duration = max(0.1, data["end"] - data["start"])
            score = (word_count * 1.3) + (avg_conf * 11.0) + (duration * 2.8)
            if last_speaker is not None and spk == last_speaker:
                score *= 1.60
            if locked_speaker is not None and spk == locked_speaker:
                score *= 2.25
            scores[spk] = score
        if not scores:
            return "", 0.0, 0, None
        best = max(scores, key=scores.get)
        data = speaker_data[best]
        text = " ".join(data["words"]).strip()
        conf = float(np.mean(data["confidences"])) if data["confidences"] else 0.0
        return text, conf, len(speaker_data), best
    except Exception:
        try:
            alt = deepgram_data["results"]["channels"][0]["alternatives"][0]
            return alt.get("transcript", "").strip(), float(alt.get("confidence", 0.0)), 1, None
        except Exception:
            return "", 0.0, 0, None

# ============================================================
# TEXT PROCESSING
# ============================================================
def to_roman_urdu(text):
    if not text:
        return text
    if re.search(r'[\u0600-\u06FF]', text):
        return unidecode(text)
    return text

def convert_numbers_to_digits(text):
    if not text:
        return text
    def replace_word(match):
        w = match.group(0)
        return Config.NUMBER_WORDS.get(w.lower(), w)
    return re.sub(r'\b\w+\b', replace_word, text)

def combine_number_multipliers(text):
    if not text:
        return text
    multipliers = [10000000, 100000, 1000, 100]
    def replace_match(m):
        return str(int(m.group(1)) * int(m.group(2)))
    for _ in range(3):
        new = re.sub(r'\b(\d+)\s+(' + '|'.join(map(str, multipliers)) + r')\b', replace_match, text)
        if new == text:
            break
        text = new
    return text

def clean_roman_urdu(text):
    if not text:
        return text
    replacements = {
        "mujhy": "mujhe", "mujhay": "mujhe", "apky": "aapke", "apki": "aapki",
        "apko": "aapko", "kry": "kare", "krna": "karna", "krain": "karein",
        "haii": "hai", "hainn": "hain", "kia": "kya", "nhi": "nahi",
        "acha": "achha", "accha": "achha", "thik": "theek",
    }
    words = [replacements.get(w.lower(), w) for w in text.strip().split()]
    return re.sub(r'\s+', ' ', " ".join(words)).strip()

# ============================================================
# DEEPGRAM
# ============================================================
def transcribe_deepgram(audio_bytes, last_speaker=None, locked_speaker=None):
    api_key = get_api_key()
    if not api_key:
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False,
                "error": "DEEPGRAM_API_KEY missing", "speaker_id": None}
    params = [
        ("model", Config.DEEPGRAM_MODEL),
        ("language", Config.DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("numerals", "true"),
        ("diarize", "true"),
        ("utterances", "true"),
    ]
    for term in Config.KEYTERMS:
        params.append(("keywords", term))
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}
    try:
        r = requests.post(Config.DEEPGRAM_URL, params=params, headers=headers,
                          data=audio_bytes, timeout=Config.DEEPGRAM_TIMEOUT)
    except requests.RequestException as e:
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False,
                "error": f"Network: {e}", "speaker_id": None}
    if r.status_code != 200:
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False,
                "error": f"Deepgram {r.status_code}", "speaker_id": None}
    data = r.json()
    text, conf, spk_count, speaker_id = pick_better_speaker(
        data, last_speaker=last_speaker, locked_speaker=locked_speaker
    )
    text = clean_roman_urdu(combine_number_multipliers(convert_numbers_to_digits(to_roman_urdu(text))))
    if not text or len(text.strip()) < 2:
        return {"text": "[No clear speech detected]", "confidence": 0.0, "speaker_count": spk_count,
                "success": True, "error": None, "speaker_id": speaker_id}
    return {"text": text, "confidence": conf, "speaker_count": spk_count,
            "success": True, "error": None, "speaker_id": speaker_id}

# ============================================================
# MAIN PROCESS
# ============================================================
def process_voice_input(audio_bytes, last_speaker=None, locked_speaker=None):
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": None, "speaker_id": None, "error": "Audio too quiet or too short."}
    result = transcribe_deepgram(cleaned["bytes"], last_speaker=last_speaker, locked_speaker=locked_speaker)
    if not result["success"]:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": cleaned["diagnostics"], "speaker_id": None, "error": result.get("error")}
    return {
        "success": True,
        "text": result["text"],
        "confidence": result["confidence"],
        "speaker_count": result["speaker_count"],
        "diagnostics": cleaned["diagnostics"],
        "speaker_id": result.get("speaker_id"),
        "error": None
    }

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎤 Voice Agent — Final")
st.caption("Soft Voice Boost + Speaker Lock + Continuity + Clean Roman Urdu")

defaults = {
    "last_text": "", "last_conf": 0.0, "last_speakers": 0,
    "last_diag": None, "last_speaker_id": None, "locked_speaker_id": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.subheader("Speakers Control")
c1, c2 = st.columns(2)
with c1:
    if st.button("Lock Current Speaker", use_container_width=True):
        if st.session_state.last_speaker_id is not None:
            st.session_state.locked_speaker_id = st.session_state.last_speaker_id
            st.success(f"Speaker {st.session_state.locked_speaker_id} locked as Target")
        else:
            st.warning("First record your voice then lock your voice")
with c2:
    if st.button("Unlock", use_container_width=True):
        st.session_state.locked_speaker_id = None
        st.info("Lock removed")

if st.session_state.locked_speaker_id is not None:
    st.success(f"Target Locked → Speaker ID: **{st.session_state.locked_speaker_id}**")

audio_output = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic"
)

if audio_output and audio_output.get("bytes"):
    with st.spinner("Processing..."):
        result = process_voice_input(
            audio_output["bytes"],
            last_speaker=st.session_state.last_speaker_id,
            locked_speaker=st.session_state.locked_speaker_id
        )
    if result["success"]:
        st.session_state.last_text = result["text"]
        st.session_state.last_conf = result["confidence"]
        st.session_state.last_speakers = result["speaker_count"]
        st.session_state.last_diag = result.get("diagnostics")
        if result.get("speaker_id") is not None:
            st.session_state.last_speaker_id = result["speaker_id"]
        st.success("Done")
    else:
        st.error(result.get("error", "Failed"))

if st.session_state.last_diag:
    d = st.session_state.last_diag
    with st.expander("Diagnostics"):
        st.write(f"SNR: **{d.get('snr_db')} dB** | Voice: **{d.get('voice_type')}** | Zone: **{d.get('snr_zone')}**")
        st.write(f"Filters: `{', '.join(d.get('filters_applied', []))}`")

st.divider()
st.subheader("Transcription")

if st.session_state.last_text:
    st.markdown(
        f"""
        <div style="background:#1e1e2e;padding:20px;border-radius:12px;color:#cdd6f4;font-size:18px;line-height:1.65;">
            {st.session_state.last_text}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"Confidence: {st.session_state.last_conf:.2f} | Speakers detected: {st.session_state.last_speakers}")
else:
    st.info("Record something to see transcription")

if st.button("Clear", use_container_width=True):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()
