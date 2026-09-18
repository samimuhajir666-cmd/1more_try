"""
VOICE-TO-TEXT AGENT - Human-Like Listening Edition
===================================================
Deepgram Nova-2 + Roman Urdu
Human-like: adaptive noise handling, smart speaker focus, clear transcription

Run: streamlit run app.py
"""

import io
import os
import re
import time
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from collections import deque
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Voice Agent",
    page_icon="🎤",
    layout="centered"
)

# ============================================================
# CONFIGURATION - Tuned for Human-Like Listening
# ============================================================
class Config:
    # --- Deepgram ---
    DEEPGRAM_URL      = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_MODEL    = "nova-2"
    DEEPGRAM_LANGUAGE = "hi-Latn"
    DEEPGRAM_TIMEOUT  = 60

    KEYTERMS = [
        "Python", "Streamlit", "Jupyter", "NumPy", "API",
        "machine learning", "function", "variable", "class",
        "Assalam", "Alaikum", "Namaz", "Salam", "Allah", "Quran",
        "Ramadan", "InshaAllah", "MashaAllah", "Alhamdulillah",
        "Shukriya", "Khuda Hafiz", "mujhe", "aapko", "karna",
        "chahiye", "hona", "theek", "acha", "nahi", "haan",
    ]

    NUMBER_WORDS = {
        "aik": "1", "ek": "1", "ik": "1",
        "do": "2", "dou": "2",
        "teen": "3", "tin": "3",
        "chaar": "4", "char": "4",
        "paanch": "5", "panch": "5",
        "chhe": "6", "chay": "6",
        "saat": "7", "aath": "8",
        "nau": "9", "das": "10",
        "gyarah": "11", "barah": "12",
        "bees": "20", "tees": "30",
        "chalees": "40", "pachaas": "50",
        "saath": "60", "sattar": "70",
        "assi": "80", "navve": "90",
        "sau": "100", "hazaar": "1000",
        "lakh": "100000", "crore": "10000000",
    }

    # --- Audio Quality Thresholds ---
    MIN_DURATION_SECONDS = 0.3
    MAX_DURATION_SECONDS = 120
    MIN_RMS_ENERGY       = 15.0    # Lower = more sensitive
    HIGHPASS_HZ          = 80

    # --- SNR Zones (Signal-to-Noise) ---
    SNR_EXCELLENT   = 30.0   # Studio quality
    SNR_GOOD        = 22.0   # Normal room
    SNR_ACCEPTABLE  = 14.0   # Noisy environment
    # Below 14 = very noisy (bus, street, crowd)

    # --- Voice Type Detection ---
    LOUD_PEAK_THRESHOLD  = 22000   # Shouting
    SOFT_RMS_MULTIPLIER  = 2.8     # Soft voice detection threshold
    FAR_RMS_MULTIPLIER   = 4.5     # Far from mic detection

    # --- Boost Settings ---
    SOFT_BOOST_DB   = 12.0   # Boost for soft voice
    FAR_BOOST_DB    = 16.0   # Boost for far voice
    VAD_FRAME_MS    = 20     # VAD frame size

    # --- Speaker Continuity ---
    # How much we favor the previous speaker (prevents speaker jumping)
    CONTINUITY_BONUS = 1.55

    # --- Pre-emphasis ---
    PRE_EMPHASIS_COEFF = 0.12   # Consonant clarity boost


# ============================================================
# API KEY
# ============================================================
def get_api_key():
    try:
        key = st.secrets.get("DEEPGRAM_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("DEEPGRAM_API_KEY")


# ============================================================
# STEP 1: AUDIO ANALYSIS
# Understand what we're dealing with before processing
# ============================================================

def estimate_snr(audio, sr):
    """
    Signal-to-Noise Ratio estimation.
    Like how well you can hear someone in a room.
    Higher = cleaner audio.
    """
    try:
        frame_len = int(sr * 0.02)  # 20ms frames
        if frame_len < 1 or len(audio) < frame_len * 2:
            return 0.0

        n_frames = len(audio) // frame_len
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len]**2) + 1e-10)
            for i in range(n_frames)
        ])

        # Noise = quietest 10% of frames
        # Signal = loudest 90th percentile
        noise  = np.percentile(energies, 10) + 1e-10
        signal = np.percentile(energies, 90) + 1e-10

        snr = 20 * np.log10(signal / noise)
        return float(np.clip(snr, 0, 80))
    except Exception:
        return 0.0


def analyze_voice_type(audio, sr):
    """
    Detect what kind of voice/environment we have.
    Returns: (voice_type, snr_zone, diagnostics)
    """
    overall_rms = np.sqrt(np.mean(audio.astype(np.float64)**2) + 1e-10)
    peak        = np.max(np.abs(audio.astype(np.float64)))
    snr         = estimate_snr(audio.astype(np.float64), sr)

    # Voice type
    if peak > Config.LOUD_PEAK_THRESHOLD:
        voice_type = "loud"
    elif overall_rms < (Config.MIN_RMS_ENERGY * Config.FAR_RMS_MULTIPLIER):
        voice_type = "far"
    elif overall_rms < (Config.MIN_RMS_ENERGY * Config.SOFT_RMS_MULTIPLIER):
        voice_type = "soft"
    else:
        voice_type = "normal"

    # SNR zone
    if snr >= Config.SNR_EXCELLENT:
        snr_zone = "excellent"
    elif snr >= Config.SNR_GOOD:
        snr_zone = "good"
    elif snr >= Config.SNR_ACCEPTABLE:
        snr_zone = "acceptable"
    else:
        snr_zone = "noisy"

    return voice_type, snr_zone, {
        "snr_db": round(snr, 1),
        "rms": round(float(overall_rms), 2),
        "peak": round(float(peak), 2),
        "voice_type": voice_type,
        "snr_zone": snr_zone,
    }


# ============================================================
# STEP 2: NOISE REDUCTION FILTERS
# ============================================================

def highpass_filter(audio, sr, cutoff=80):
    """Remove low-frequency rumble (fan hum, AC noise, wind)"""
    try:
        nyq = 0.5 * sr
        if cutoff / nyq >= 1.0:
            return audio
        b, a = signal.butter(4, cutoff / nyq, btype='high')
        return signal.filtfilt(b, a, audio.astype(np.float64))
    except Exception:
        return audio


def spectral_noise_reduction(audio, sr, strength=1.5):
    """
    Spectral subtraction noise reduction.
    Estimates noise floor and subtracts it from signal.
    Like noise-cancelling headphones.
    """
    try:
        audio_f = audio.astype(np.float64)
        f, t, Zxx = signal.stft(audio_f, fs=sr, nperseg=512, noverlap=384)
        mag   = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Estimate noise from quietest frames
        frame_energy = np.mean(mag**2, axis=0)
        noise_mask   = frame_energy < np.percentile(frame_energy, 12)

        if np.sum(noise_mask) > 0:
            noise_profile = np.mean(mag[:, noise_mask], axis=1, keepdims=True)
        else:
            noise_profile = np.min(mag, axis=1, keepdims=True) * 0.5

        # Subtract noise (with floor to prevent musical noise)
        mag_clean = np.maximum(
            mag - strength * noise_profile,
            0.08 * mag  # 8% floor - prevents complete silence artifacts
        )

        # Reconstruct
        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, audio_clean = signal.istft(Zxx_clean, fs=sr, nperseg=512, noverlap=384)

        # Match length
        if len(audio_clean) < len(audio_f):
            audio_clean = np.pad(audio_clean, (0, len(audio_f) - len(audio_clean)))
        else:
            audio_clean = audio_clean[:len(audio_f)]

        return audio_clean
    except Exception:
        return audio


# ============================================================
# STEP 3: VOICE ENHANCEMENT
# Make the voice louder/clearer based on type
# ============================================================

def pre_emphasis_filter(audio, coeff=None):
    """
    Boost high frequencies (consonants).
    Consonants carry meaning - they fade first with distance.
    Like turning up treble on a voice.
    """
    coeff = Config.PRE_EMPHASIS_COEFF if coeff is None else coeff
    try:
        audio = audio.astype(np.float64)
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    except Exception:
        return audio


def smart_vad_boost(audio, sr, boost_db=None):
    """
    Voice Activity Detection + Selective Boost.
    
    Human-like: boosts ONLY speech frames, not silence/noise.
    Result: voice gets louder, background stays quiet.
    """
    boost_db  = boost_db or Config.SOFT_BOOST_DB
    frame_ms  = Config.VAD_FRAME_MS
    try:
        audio     = audio.astype(np.float64)
        frame_len = int(sr * frame_ms / 1000)
        if frame_len < 1 or len(audio) < frame_len * 2:
            return audio

        n_frames = len(audio) // frame_len
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len]**2) + 1e-10)
            for i in range(n_frames)
        ])

        # Dynamic threshold: adapts to the audio's own noise floor
        noise_floor = np.percentile(energies, 12)
        threshold   = max(noise_floor * 1.6, Config.MIN_RMS_ENERGY / 32767.0 * 0.4)
        speech_mask = energies > threshold

        speech_ratio = np.mean(speech_mask)
        if speech_ratio < 0.02:  # Almost no speech - don't boost noise
            return audio

        boost   = 10 ** (boost_db / 20)
        out     = audio.copy()

        for i in range(n_frames):
            if speech_mask[i]:
                out[i*frame_len:(i+1)*frame_len] *= boost

        # Normalize after boost
        peak = np.max(np.abs(out)) + 1e-8
        if peak > 32767 * 0.9:
            out = out / peak * 32767 * 0.88

        return out
    except Exception:
        return audio


def dynamic_range_compressor(audio, sr, frame_ms=20):
    """
    Per-frame compression.
    
    Human-like: automatically adjusts volume moment-to-moment.
    Quiet parts get louder, loud parts stay controlled.
    Like automatic volume control on a human ear.
    """
    try:
        audio     = audio.astype(np.float64)
        frame_len = max(1, int(sr * frame_ms / 1000))
        n_frames  = len(audio) // frame_len
        if n_frames < 2:
            return audio

        target_rms = Config.MIN_RMS_ENERGY * 4.0  # Target comfortable listening level
        max_gain   = 10 ** (14.0 / 20)  # Max 14dB gain
        out        = audio.copy()
        prev_gain  = 1.0
        smoothing  = 0.3  # Smooth gain changes (prevent pumping)

        for i in range(n_frames):
            seg = audio[i*frame_len:(i+1)*frame_len]
            rms = np.sqrt(np.mean(seg**2) + 1e-8)

            if rms < 8.0:  # Near-silence: no gain
                gain = 1.0
            else:
                gain = float(np.clip(target_rms / rms, 1.0, max_gain))

            # Smooth the gain transition
            gain      = smoothing * prev_gain + (1 - smoothing) * gain
            prev_gain = gain
            out[i*frame_len:(i+1)*frame_len] = seg * gain

        # Safety clip
        peak = np.max(np.abs(out)) + 1e-8
        if peak > 31000:
            out = out / peak * 31000

        return out
    except Exception:
        return audio


def agc_peak_limiter(audio, sr, target_db=-2.0):
    """
    Automatic Gain Control + Peak Limiter.
    Prevents clipping for loud voices.
    """
    try:
        audio     = audio.astype(np.float64)
        peak      = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        audio_norm = audio / peak
        rms        = np.sqrt(np.mean(audio_norm**2) + 1e-10)
        if rms < 1e-10:
            return audio
        current_db = 20 * np.log10(rms)
        gain_db    = np.clip(target_db - current_db, -20, 18)
        gain       = 10 ** (gain_db / 20)
        audio_agc  = audio_norm * gain
        # Soft limiter (tanh)
        audio_lim  = np.tanh(audio_agc / 0.95) * 0.95
        return (audio_lim * 32767).astype(np.float64)
    except Exception:
        return audio


def normalize_audio(audio, target_peak=0.92):
    """Final normalization to consistent output level"""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak * target_peak * 32767


def is_silent(audio, rms_threshold=0.8):
    if audio is None or len(audio) == 0:
        return True
    return np.sqrt(np.mean(audio.astype(np.float64)**2) + 1e-10) < rms_threshold


# ============================================================
# STEP 4: ADAPTIVE PIPELINE
# Choose the right processing based on voice type
# Like a smart sound engineer
# ============================================================

def adaptive_preprocess(audio, sr):
    """
    Human-like adaptive processing:
    
    - LOUD voice   → Limit peaks, normalize
    - SOFT voice   → Boost speech frames (VAD), not noise  
    - FAR voice    → Pre-emphasis + compression + aggressive boost
    - NORMAL voice → Light noise reduction only
    - NOISY env    → Spectral noise reduction first
    - CRITICAL env → Aggressive NR + all processing
    """
    audio = audio.astype(np.float64)
    voice_type, snr_zone, analysis = analyze_voice_type(audio, sr)
    filters = []

    # ── Step A: Noise Reduction (based on environment) ──
    if snr_zone == "excellent":
        audio = highpass_filter(audio, sr, 80)
        filters.append("hp80")

    elif snr_zone == "good":
        audio = highpass_filter(audio, sr, 90)
        audio = spectral_noise_reduction(audio, sr, strength=1.2)
        filters.append("hp90 + NR(light)")

    elif snr_zone == "acceptable":
        audio = highpass_filter(audio, sr, 100)
        audio = spectral_noise_reduction(audio, sr, strength=1.8)
        filters.append("hp100 + NR(medium)")

    else:  # noisy
        audio = highpass_filter(audio, sr, 120)
        audio = spectral_noise_reduction(audio, sr, strength=2.5)
        filters.append("hp120 + NR(aggressive)")

    # ── Step B: Voice Enhancement (based on voice type) ──
    if voice_type == "loud":
        audio = agc_peak_limiter(audio, sr)
        filters.append("agc_limiter")

    elif voice_type == "far":
        # Far voice needs: consonant boost + compression + hard boost
        audio = pre_emphasis_filter(audio)
        audio = dynamic_range_compressor(audio, sr)
        audio = smart_vad_boost(audio, sr, boost_db=Config.FAR_BOOST_DB)
        filters.append("pre_emphasis + compressor + far_boost")

    elif voice_type == "soft":
        # Soft voice: boost speech frames selectively
        audio = smart_vad_boost(audio, sr, boost_db=Config.SOFT_BOOST_DB)
        filters.append("soft_boost(VAD)")

    else:  # normal
        # Normal: just light compression for consistency
        audio = dynamic_range_compressor(audio, sr)
        filters.append("light_compressor")

    # ── Step C: Final Normalization ──
    audio = normalize_audio(audio)

    analysis["filters_applied"] = filters
    return np.clip(audio, -32768, 32767).astype(np.int16), analysis


def preprocess_audio(audio_bytes):
    """Main audio preprocessing entry point"""
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if sr <= 0:
            return None
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio    = audio.astype(np.float64)
        duration = len(audio) / sr

        if duration < Config.MIN_DURATION_SECONDS:
            return None
        if is_silent(audio):
            return None
        if duration > Config.MAX_DURATION_SECONDS:
            audio    = audio[:int(Config.MAX_DURATION_SECONDS * sr)]
            duration = len(audio) / sr

        audio_clean, analysis = adaptive_preprocess(audio, sr)

        buf = io.BytesIO()
        wav.write(buf, sr, audio_clean)
        buf.seek(0)

        return {
            "bytes":    buf.read(),
            "duration": duration,
            "sr":       sr,
            "analysis": analysis,
        }
    except Exception:
        return None


# ============================================================
# STEP 5: FALLBACK - Light processing for retry
# If full processing distorts audio, retry with minimal
# ============================================================

def light_preprocess(audio_bytes):
    """Minimal processing - just clean without boosting"""
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        audio = highpass_filter(audio, sr, 80)
        audio = normalize_audio(audio)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        buf   = io.BytesIO()
        wav.write(buf, sr, audio)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ============================================================
# STEP 6: SMART SPEAKER SELECTION
# When multiple speakers detected, pick the right one
# Human-like: focuses on main speaker, not just loudest
# ============================================================

def pick_main_speaker(deepgram_data, last_speaker=None):
    """
    Human-like speaker selection.
    
    Score each speaker based on:
    1. Word count (more words = likely main speaker)
    2. Confidence (higher = clearer speech)  
    3. Duration (longer speaking = likely main speaker)
    4. Continuity (same as last turn = bonus)
    
    Unlike naive approach: does NOT just pick loudest.
    Loud background interrupter gets PENALIZED vs continuous speaker.
    """
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

        # Group words by speaker
        speaker_data = {}
        for w in words:
            spk = w.get("speaker", 0)
            if spk not in speaker_data:
                speaker_data[spk] = {
                    "words":       [],
                    "confidences": [],
                    "start":       w.get("start", 0),
                    "end":         w.get("end",   0),
                }
            speaker_data[spk]["words"].append(w.get("word", ""))
            speaker_data[spk]["confidences"].append(float(w.get("confidence", 0.0)))
            speaker_data[spk]["end"] = w.get("end", speaker_data[spk]["end"])

        # Score each speaker
        scores = {}
        for spk, data in speaker_data.items():
            word_count = len(data["words"])
            avg_conf   = float(np.mean(data["confidences"])) if data["confidences"] else 0.0
            duration   = max(0.1, data["end"] - data["start"])
            words_per_sec = word_count / max(duration, 0.1)

            # Weighted score
            # - Confidence is most important (clear speech > loud speech)
            # - Duration favors continuous speakers
            # - Words/sec: natural speech rate is 2-4 words/sec
            score = (
                avg_conf   * 12.0 +
                word_count *  1.5 +
                duration   *  3.0 +
                min(words_per_sec, 5.0) * 1.0  # Cap rate bonus (screaming ≠ main speaker)
            )

            # Continuity bonus: same speaker as last turn
            if last_speaker is not None and spk == last_speaker:
                score *= Config.CONTINUITY_BONUS

            scores[spk] = score

        if not scores:
            return "", 0.0, 0, None

        # Pick highest scoring speaker
        best_spk  = max(scores, key=scores.get)
        best_data = speaker_data[best_spk]
        text      = " ".join(best_data["words"]).strip()
        conf      = float(np.mean(best_data["confidences"])) if best_data["confidences"] else 0.0

        return text, conf, len(speaker_data), best_spk

    except Exception:
        try:
            alt  = deepgram_data["results"]["channels"][0]["alternatives"][0]
            text = alt.get("transcript", "").strip()
            conf = float(alt.get("confidence", 0.0) or 0.0)
            return text, conf, 1, None
        except Exception:
            return "", 0.0, 0, None


# ============================================================
# STEP 7: TEXT PROCESSING - Roman Urdu Cleanup
# ============================================================

def to_roman_urdu(text):
    """Convert Urdu script to Roman if mixed"""
    if not text:
        return text
    if re.search(r'[\u0600-\u06FF]', text):
        return unidecode(text)
    return text


def convert_numbers(text):
    """Roman Urdu number words → digits"""
    if not text:
        return text

    def replace_word(match):
        w = match.group(0)
        return Config.NUMBER_WORDS.get(w.lower(), w)

    text = re.sub(r'\b\w+\b', replace_word, text)

    # Combine: "2 1000" → "2000"
    multipliers = [10000000, 100000, 1000, 100]
    for _ in range(3):
        new = re.sub(
            r'\b(\d+)\s+(' + '|'.join(map(str, multipliers)) + r')\b',
            lambda m: str(int(m.group(1)) * int(m.group(2))),
            text
        )
        if new == text:
            break
        text = new

    return text


def normalize_roman_urdu(text):
    """
    Normalize common Roman Urdu spelling variations.
    People write the same word many ways - standardize it.
    """
    if not text:
        return text

    # Common spelling normalizations
    replacements = {
        # Pronouns
        "mujhy": "mujhe",   "mujhay": "mujhe",
        "apky":  "aapke",   "apki":   "aapki",   "apko": "aapko",
        "humko": "hamko",   "humein": "hamein",
        "unko":  "unko",    "unhe":   "unhe",
        # Verbs
        "kry":   "kare",    "krna":   "karna",   "krain": "karein",
        "ho":    "ho",      "hona":   "hona",    "hoga":  "hoga",
        "tha":   "tha",     "thi":    "thi",     "thy":   "the",
        # Common words
        "haii":  "hai",     "hainn":  "hain",
        "kia":   "kya",
        "nhi":   "nahi",    "nai":    "nahi",
        "acha":  "achha",   "accha":  "achha",
        "thik":  "theek",   "tik":    "theek",
        "phir":  "phir",    "fer":    "phir",
        "ab":    "ab",      "abb":    "ab",
        "bhi":   "bhi",     "bi":     "bhi",
        "woh":   "woh",     "wo":     "woh",     "vo": "woh",
        "yeh":   "yeh",     "ye":     "yeh",
        "kuch":  "kuch",    "kchh":   "kuch",
        "bohat": "bohot",   "bahut":  "bohot",
        "bilkul":"bilkul",  "bilkl":  "bilkul",
        # Greetings
        "assalam": "assalam", "salam": "salam",
        "alaikum": "alaikum",
    }

    words   = text.strip().split()
    cleaned = []
    for w in words:
        lower = w.lower()
        if lower in replacements:
            cleaned.append(replacements[lower])
        else:
            cleaned.append(w)

    return re.sub(r'\s+', ' ', " ".join(cleaned)).strip()


def postprocess_text(text):
    """Full text processing pipeline"""
    text = to_roman_urdu(text)
    text = convert_numbers(text)
    text = normalize_roman_urdu(text)
    return text


# ============================================================
# STEP 8: DEEPGRAM TRANSCRIPTION
# ============================================================

def transcribe_deepgram(audio_bytes, last_speaker=None):
    api_key = get_api_key()
    if not api_key:
        return {
            "text": "", "confidence": 0.0, "speaker_count": 0,
            "speaker_id": None, "success": False,
            "error": "DEEPGRAM_API_KEY missing"
        }

    params = [
        ("model",        Config.DEEPGRAM_MODEL),
        ("language",     Config.DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate",    "true"),
        ("numerals",     "true"),
        ("diarize",      "true"),
        ("utterances",   "true"),
    ]
    for term in Config.KEYTERMS:
        params.append(("keywords", term))

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type":  "audio/wav",
    }

    try:
        r = requests.post(
            Config.DEEPGRAM_URL, params=params,
            headers=headers, data=audio_bytes,
            timeout=Config.DEEPGRAM_TIMEOUT
        )
    except requests.RequestException as e:
        return {
            "text": "", "confidence": 0.0, "speaker_count": 0,
            "speaker_id": None, "success": False,
            "error": f"Network error: {e}"
        }

    if r.status_code != 200:
        return {
            "text": "", "confidence": 0.0, "speaker_count": 0,
            "speaker_id": None, "success": False,
            "error": f"Deepgram error {r.status_code}: {r.text[:200]}"
        }

    data               = r.json()
    text, conf, n_spk, spk_id = pick_main_speaker(data, last_speaker=last_speaker)
    text               = postprocess_text(text)

    if not text or len(text.strip()) < 2:
        return {
            "text": "[No clear speech detected]",
            "confidence": 0.0, "speaker_count": n_spk,
            "speaker_id": spk_id, "success": True, "error": None
        }

    return {
        "text": text, "confidence": conf,
        "speaker_count": n_spk, "speaker_id": spk_id,
        "success": True, "error": None
    }


# ============================================================
# STEP 9: MAIN PROCESSING PIPELINE
# ============================================================

def process_voice(audio_bytes, last_speaker=None):
    """
    Complete human-like processing pipeline:
    1. Preprocess (adaptive noise + enhancement)
    2. Transcribe (Deepgram)
    3. Retry with light processing if full processing distorted
    4. Return clean result
    """
    # Full adaptive processing
    processed = preprocess_audio(audio_bytes)
    if processed is None:
        return {
            "success":       False,
            "text":          "",
            "confidence":    0.0,
            "speaker_count": 0,
            "speaker_id":    None,
            "analysis":      None,
            "error":         "Audio too quiet or too short. Please speak again.",
        }

    result = transcribe_deepgram(processed["bytes"], last_speaker=last_speaker)

    # If full processing produced nothing → retry with light processing
    # (prevents aggressive NR from destroying a clean recording)
    if result.get("text") == "[No clear speech detected]":
        light_bytes = light_preprocess(audio_bytes)
        if light_bytes:
            retry = transcribe_deepgram(light_bytes, last_speaker=last_speaker)
            if retry.get("success") and retry.get("text") != "[No clear speech detected]":
                processed["analysis"]["filters_applied"].append("⚡ retried_minimal")
                return {
                    "success":       True,
                    "text":          retry["text"],
                    "confidence":    retry["confidence"],
                    "speaker_count": retry["speaker_count"],
                    "speaker_id":    retry["speaker_id"],
                    "analysis":      processed["analysis"],
                    "error":         None,
                }

    return {
        "success":       result["success"],
        "text":          result.get("text", ""),
        "confidence":    result.get("confidence", 0.0),
        "speaker_count": result.get("speaker_count", 0),
        "speaker_id":    result.get("speaker_id"),
        "analysis":      processed["analysis"],
        "error":         result.get("error"),
    }


# ============================================================
# HUMAN FEEDBACK MESSAGES
# Tell user in plain language what happened
# ============================================================

def get_human_feedback(analysis):
    """Plain language feedback - like a real sound engineer would say"""
    if not analysis:
        return None, "default"

    vt   = analysis.get("voice_type",  "normal")
    zone = analysis.get("snr_zone",    "good")
    snr  = analysis.get("snr_db",      0)

    if vt == "loud":
        return "🔊 Loud voice — controlled automatically.", "success"
    if vt == "far":
        return (
            f"📏 Sounds like you're far from the mic (SNR: {snr}dB). "
            "I boosted it, but moving closer will improve accuracy.",
            "warning"
        )
    if vt == "soft":
        return "🔈 Soft voice — boosted for clarity.", "info"
    if zone == "noisy":
        return (
            f"🌪️ Very noisy environment (SNR: {snr}dB). "
            "I cleaned it up — a quieter spot works better.",
            "warning"
        )
    if zone == "acceptable":
        return f"🌤️ Some background noise (SNR: {snr}dB) — cleaned up.", "info"

    return f"✅ Clear audio (SNR: {snr}dB) — perfect conditions.", "success"


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎤 Voice Agent")
st.caption("Human-like listening: adapts to your voice & environment automatically")

# Session state
_defaults = {
    "last_text":       "",
    "last_conf":       0.0,
    "last_speakers":   0,
    "last_analysis":   None,
    "last_speaker_id": None,
    "history":         [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# API Key check
if not get_api_key():
    st.error(
        "**DEEPGRAM_API_KEY missing!**\n\n"
        "Add to `.env` file:\n```\nDEEPGRAM_API_KEY=your_key_here\n```\n\n"
        "Or in Streamlit Cloud → Settings → Secrets"
    )
    st.stop()

# ── Recording ──
st.markdown("### 🎙️ Record")
audio_output = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic",
)

# ── Process ──
if audio_output and audio_output.get("bytes"):
    with st.spinner("🧠 Listening & processing..."):
        result = process_voice(
            audio_output["bytes"],
            last_speaker=st.session_state.last_speaker_id,
        )

    if result["success"]:
        st.session_state.last_text       = result["text"]
        st.session_state.last_conf       = result["confidence"]
        st.session_state.last_speakers   = result["speaker_count"]
        st.session_state.last_analysis   = result.get("analysis")
        if result.get("speaker_id") is not None:
            st.session_state.last_speaker_id = result["speaker_id"]

        # Add to history
        if result["text"] and result["text"] != "[No clear speech detected]":
            st.session_state.history.append({
                "text": result["text"],
                "conf": result["confidence"],
                "time": time.strftime("%H:%M:%S"),
            })

        st.success("✅ Done!")
    else:
        st.session_state.last_analysis = result.get("analysis")
        st.error(f"❌ {result.get('error', 'Processing failed')}")

# ── Human Feedback ──
if st.session_state.last_analysis:
    msg, msg_type = get_human_feedback(st.session_state.last_analysis)
    if msg:
        if msg_type == "success":
            st.success(msg)
        elif msg_type == "warning":
            st.warning(msg)
        else:
            st.info(msg)

# ── Transcription Display ──
st.markdown("---")
st.markdown("### 📝 Transcription")

if st.session_state.last_text:
    # Main transcription box
    st.markdown(
        f"""
        <div style="
            background: #1e1e2e;
            padding: 20px 24px;
            border-radius: 12px;
            color: #cdd6f4;
            font-size: 19px;
            line-height: 1.7;
            border-left: 4px solid #89b4fa;
        ">
            {st.session_state.last_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stats row
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{st.session_state.last_conf:.0%}")
    col2.metric("Speakers",   st.session_state.last_speakers)
    col3.metric("History",    len(st.session_state.history))

else:
    st.info("🎙️ Record something to see transcription here.")

# ── Technical Details (collapsible) ──
if st.session_state.last_analysis:
    a = st.session_state.last_analysis
    with st.expander("🔧 Technical Details", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("SNR",        f"{a.get('snr_db', 0)} dB")
        col2.metric("Voice Type", a.get("voice_type", "?").title())

        st.write(f"**SNR Zone:** {a.get('snr_zone', '?').title()}")
        st.write(f"**Filters Applied:**")
        for f in a.get("filters_applied", []):
            st.write(f"  → `{f}`")

# ── History ──
if len(st.session_state.history) > 1:
    with st.expander(f"📜 Session History ({len(st.session_state.history)} entries)", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.markdown(
                f"**{entry['time']}** _(conf: {entry['conf']:.0%})_\n\n"
                f"> {entry['text']}"
            )
            if i < len(st.session_state.history):
                st.markdown("---")

# ── Clear Button ──
if st.button("🗑️ Clear All", use_container_width=True):
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.rerun()

# ── Footer ──
st.markdown("---")
st.caption(
    "🧠 Human-like processing: "
    "adaptive noise reduction · smart speaker focus · "
    "Roman Urdu normalization · auto-retry · session history"
)
