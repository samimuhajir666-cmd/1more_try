"""
VOICE-TO-TEXT AGENT - Production Ready
Deepgram Nova-2 + Roman Urdu + Adaptive SNR Filtering
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

    NUMBER_WORDS = {
        "aik": "1", "ek": "1", "ik": "1",
        "do": "2", "dou": "2", "teen": "3", "tin": "3",
        "chaar": "4", "char": "4", "paanch": "5", "panch": "5",
        "chhe": "6", "chay": "6", "che": "6",
        "saat": "7", "sath": "7", "aath": "8", "ath": "8",
        "nau": "9", "no": "9", "das": "10", "dus": "10",
        "bees": "20", "bis": "20", "tees": "30", "tis": "30",
        "chalees": "40", "pachaas": "50", "saath": "60",
        "sattar": "70", "assi": "80", "navve": "90",
        "sau": "100", "so": "100",
        "hazaar": "1000", "lakh": "100000", "crore": "10000000",
    }

    # SNR thresholds (proven science)
    SNR_SAFE_MIN     = 30.0
    SNR_DEGRADED_MIN = 18.0

    # Wind detection (simple, effective)
    WIND_ZCR_THRESHOLD = 0.15

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
# API KEY
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
# FEATURE 1: AGC + PEAK LIMITER (amplitude-based, not pitch)
# ============================================================
def apply_agc_peak_limiter(audio, sr, target_db=None, threshold=None):
    """Clamp loud audio based on AMPLITUDE, not pitch."""
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
# FEATURE 2: SMART VAD + BOOST (amplitude-based)
# ============================================================
def smart_vad_boost(audio, sr, boost_db=None, frame_ms=None):
    """Boost soft voice based on AMPLITUDE, not pitch."""
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
# HELPER: FILTERS
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


# ============================================================
# DIAGNOSTIC: SNR ESTIMATION (for adaptive filtering)
# ============================================================
def estimate_snr(audio, sr):
    """Estimate SNR from frame energy percentiles."""
    try:
        audio = audio.astype(np.float64)
        frame_len = int(sr * 0.02)
        if frame_len < 1:
            return 0.0
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return 0.0
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len] ** 2) + 1e-10)
            for i in range(n_frames)
        ])
        noise = np.percentile(energies, 10) + 1e-10
        signal_p = np.percentile(energies, 90) + 1e-10
        return float(np.clip(20 * np.log10(signal_p / noise), 0, 80))
    except Exception:
        return 0.0


# ============================================================
# DIAGNOSTIC: WIND DETECTION
# ============================================================
def detect_wind_noise(audio, sr):
    try:
        zcr = np.mean(np.abs(np.diff(np.sign(audio.astype(np.float64))))) / 2
        return bool(zcr > Config.WIND_ZCR_THRESHOLD)
    except Exception:
        return False


# ============================================================
# DIAGNOSTIC: PITCH ESTIMATION (display only, not for filtering)
# ============================================================
def estimate_pitch(audio, sr, fmin=80, fmax=400, max_seconds=10):
    """Estimate pitch for DISPLAY ONLY. Do not use for filtering decisions."""
    try:
        audio = audio.astype(np.float64)
        max_samples = int(max_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        audio = audio - np.mean(audio)
        if len(audio) > 1:
            audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        frame_len = int(sr * 0.04)
        hop = int(sr * 0.02)
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)

        pitches = []
        for start in range(0, len(audio) - frame_len, hop * 5):
            frame = audio[start:start + frame_len]
            if np.sqrt(np.mean(frame ** 2)) < 100:
                continue
            n = len(frame)
            fft_size = 1 << (2 * n - 1).bit_length()
            fft = np.fft.rfft(frame, fft_size)
            corr = np.fft.irfft(fft * np.conj(fft))[:max_lag + 1]
            corr = corr / (corr[0] + 1e-10)
            if max_lag >= len(corr):
                continue
            segment = corr[min_lag:max_lag]
            if len(segment) == 0:
                continue
            peak_idx = np.argmax(segment) + min_lag
            if corr[peak_idx] > 0.3:
                p = sr / peak_idx
                if fmin <= p <= fmax:
                    pitches.append(p)
        return float(np.median(pitches)) if pitches else 0.0
    except Exception:
        return 0.0


# ============================================================
# SPECTRAL NOISE REDUCTION
# ============================================================
def spectral_noise_reduce(audio, sr, aggressive=False):
    try:
        audio_f = audio.astype(np.float64)
        f, t, Zxx = signal.stft(audio_f, fs=sr, nperseg=512)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        frame_energy = np.mean(mag ** 2, axis=0)
        noise_frames = frame_energy < np.percentile(frame_energy, 10)
        if np.sum(noise_frames) > 0:
            noise_profile = np.mean(mag[:, noise_frames], axis=1, keepdims=True)
        else:
            noise_profile = np.min(mag, axis=1, keepdims=True)
        alpha = 2.0 if aggressive else 1.5
        mag_clean = np.maximum(mag - alpha * noise_profile, 0.1 * mag)
        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, audio_clean = signal.istft(Zxx_clean, fs=sr)
        if len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
        else:
            audio_clean = audio_clean[:len(audio)]
        return audio_clean
    except Exception:
        return audio


# ============================================================
# ADAPTIVE PREPROCESSING (SNR-driven, amplitude-based)
# ============================================================
def adaptive_preprocess(audio, sr):
    """
    Filtering decisions based on SNR and amplitude - NOT pitch.
    Returns (processed_audio, diagnostics).
    """
    audio = audio.astype(np.float64)

    # Measure
    snr = estimate_snr(audio, sr)
    wind = detect_wind_noise(audio, sr)
    pitch = estimate_pitch(audio, sr)   # display only

    # Amplitude-based voice detection (not pitch)
    overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    peak = np.max(np.abs(audio))
    is_loud = peak > 25000               # near int16 max
    is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 1.5)

    if is_loud:
        voice_type = "loud"
    elif is_soft:
        voice_type = "soft"
    else:
        voice_type = "normal"

    # SNR zone
    if snr >= Config.SNR_SAFE_MIN:
        snr_zone = "safe"
    elif snr >= Config.SNR_DEGRADED_MIN:
        snr_zone = "degraded"
    else:
        snr_zone = "critical"

    diagnostics = {
        "pitch_hz": round(pitch, 1),
        "snr_db": round(snr, 1),
        "voice_type": voice_type,
        "snr_zone": snr_zone,
        "wind_detected": wind,
        "filters_applied": [],
    }

    # ---- SNR-based filtering ----
    if snr_zone == "safe":
        audio = highpass_filter(audio, sr, 80)
        diagnostics["filters_applied"].append("highpass_80Hz")
    elif snr_zone == "degraded":
        audio = highpass_filter(audio, sr, 100)
        audio = spectral_noise_reduce(audio, sr, aggressive=False)
        diagnostics["filters_applied"].append("highpass_100Hz")
        diagnostics["filters_applied"].append("noise_reduction")
    else:  # critical
        audio = highpass_filter(audio, sr, 120)
        audio = spectral_noise_reduce(audio, sr, aggressive=True)
        diagnostics["filters_applied"].append("highpass_120Hz")
        diagnostics["filters_applied"].append("aggressive_NR")

    # ---- Wind ----
    if wind:
        audio = highpass_filter(audio, sr, 150)
        diagnostics["filters_applied"].append("wind_filter_150Hz")

    # ---- Amplitude-based processing ----
    if is_loud:
        audio = apply_agc_peak_limiter(audio, sr)
        diagnostics["filters_applied"].append("agc_limiter")
    elif is_soft:
        audio = smart_vad_boost(audio, sr)
        diagnostics["filters_applied"].append("vad_boost")

    audio = normalize_audio(audio)
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    return audio, diagnostics


# ============================================================
# MAIN PIPELINE
# ============================================================
def preprocess_audio(audio_bytes):
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

        audio_clean, diagnostics = adaptive_preprocess(audio, sr)

        buf = io.BytesIO()
        wav.write(buf, sr, audio_clean)
        buf.seek(0)

        return {
            "bytes": buf.read(),
            "duration": duration,
            "sample_rate": int(sr),
            "diagnostics": diagnostics,
        }
    except Exception:
        return None


# ============================================================
# SPEAKER ISOLATION
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
        word = match.group(0)
        if word.lower() in Config.NUMBER_WORDS:
            return Config.NUMBER_WORDS[word.lower()]
        return word

    return re.sub(r'\b\w+\b', replace_word, text)


def combine_number_multipliers(text):
    if not text:
        return text
    multipliers = [10000000, 100000, 1000, 100]

    def replace_match(match):
        return str(int(match.group(1)) * int(match.group(2)))

    for _ in range(3):
        new_text = re.sub(
            r'\b(\d+)\s+(' + '|'.join(str(m) for m in multipliers) + r')\b',
            replace_match, text
        )
        if new_text == text:
            break
        text = new_text
    return text


def classify_non_speech(audio_bytes):
    """Classify empty-audio responses (music/noise/silence)."""
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        if rms < 5.0:
            return "[Silence - no audio detected]"
        try:
            _, psd = signal.welch(audio, fs=sr, nperseg=1024)
            psd_safe = np.maximum(psd, 1e-12)
            flatness = np.exp(np.mean(np.log(psd_safe))) / np.mean(psd_safe)
            if flatness > 0.5:
                return "[Background noise - no speech detected]"
            elif flatness < 0.15 and rms > 500:
                return "[Music playing - no speech detected]"
            else:
                return "[Non-speech audio - no words detected]"
        except Exception:
            return "[No speech detected]"
    except Exception:
        return "[No speech detected]"


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

    text = to_roman_urdu(text)
    text = convert_numbers_to_digits(text)
    text = combine_number_multipliers(text)

    if not text or len(text.strip()) < 2:
        kind = classify_non_speech(audio_bytes)
        return {"text": kind, "confidence": 0.0, "speaker_count": spk_count,
                "success": True, "error": None}

    return {"text": text, "confidence": conf, "speaker_count": spk_count,
            "success": True, "error": None}


# ============================================================
# MAIN ENTRY
# ============================================================
def process_voice_input(audio_bytes):
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0,
                "speaker_count": 0, "diagnostics": None,
                "error": "Audio too quiet or too short. Please speak clearly."}

    result = transcribe_deepgram(cleaned["bytes"], isolate_speaker=True)

    if not result["success"]:
        err = result.get("error") or "No clear speech detected."
        return {"success": False, "text": "", "confidence": 0.0,
                "speaker_count": 0, "diagnostics": cleaned["diagnostics"],
                "error": err}

    return {"success": True, "text": result["text"],
            "confidence": result["confidence"],
            "speaker_count": result["speaker_count"],
            "diagnostics": cleaned["diagnostics"], "error": None}


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
if "last_diag" not in st.session_state:
    st.session_state.last_diag = None

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
        st.session_state.last_diag = result.get("diagnostics")
        st.success("Transcription complete!")
    else:
        st.session_state.last_diag = result.get("diagnostics")
        st.error(f"Failed: {result['error']}")

# Diagnostics panel (info only)
if st.session_state.last_diag:
    d = st.session_state.last_diag
    with st.expander("🔬 Voice Diagnostics (info only)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pitch", f"{d.get('pitch_hz', 0):.0f} Hz")
        with col2:
            st.metric("SNR", f"{d.get('snr_db', 0):.1f} dB")
        with col3:
            st.metric("Volume", d.get("voice_type", "?").title())

        st.write(f"• SNR Zone: `{d.get('snr_zone', '?')}`")
        st.write(f"• Wind: {'✅ Detected' if d.get('wind_detected') else '❌ No'}")
        st.write(f"• Filters: `{', '.join(d.get('filters_applied', [])) or 'none'}`")

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
    st.session_state.last_diag = None
    st.rerun()
