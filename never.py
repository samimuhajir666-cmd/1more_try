"""
VOICE-TO-TEXT AGENT - Optimized for Transcription Quality
Deepgram Nova-2 + Enhanced Roman Urdu + Adaptive Processing
"""

import io
import os
import re
import requests
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from unidecode import unidecode

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Voice Agent Pro", page_icon="🎤", layout="centered")

# ============================================================
# CONFIG (Optimized for better transcription)
# ============================================================
class Config:
    DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
    DEEPGRAM_MODEL = "nova-2"
    DEEPGRAM_LANGUAGE = "hi-Latn"  # Best for Roman Urdu
    DEEPGRAM_TIMEOUT = 60

    KEYTERMS = [
        "Python", "Streamlit", "Jupyter", "Matplotlib", "Plotly",
        "NumPy", "API", "machine learning", "function", "variable",
        "Assalam", "Alaikum", "Namaz", "Salam", "Allah", "Quran",
        "Ramadan", "InshaAllah", "MashaAllah", "Alhamdulillah",
        "Shukriya", "Khuda Hafiz", "Allah Hafiz", "hello", "assalamoalaikum",
        "wa alaikum assalam", "how are you", "thank you", "please", "okay"
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

    # Optimized audio thresholds
    SNR_SAFE_MIN = 25.0       # Lowered for better sensitivity
    SNR_DEGRADED_MIN = 15.0   # More tolerant of noise
    MIN_RMS_ENERGY = 18.0     # Lower threshold for quiet voices
    SOFT_BOOST_DB = 14.0      # Increased boost for soft voices
    VAD_FRAME_MS = 20         # Shorter frames for better responsiveness
    HIGHPASS_HZ = 60         # Lower cutoff to preserve voice frequencies
    MIN_DURATION_SECONDS = 0.3
    MAX_DURATION_SECONDS = 120

    # Improved far-field handling
    FAR_FIELD_RMS_MULT = 2.8  # More sensitive to far-field voices
    COMPRESSOR_TARGET_RATIO = 0.25
    COMPRESSOR_MAX_GAIN_DB = 14.0  # Increased max gain
    COMPRESSOR_SMOOTHING = 0.4
    PRE_EMPHASIS_COEFF = 0.12   # More subtle pre-emphasis

# ============================================================
# API KEY
# ============================================================
def get_api_key():
    try:
        return st.secrets.get("DEEPGRAM_API_KEY")
    except Exception:
        return os.getenv("DEEPGRAM_API_KEY")

# ============================================================
# ENHANCED AUDIO PROCESSING
# ============================================================
def apply_agc_peak_limiter(audio, sr, target_db=-4.0, threshold=0.9):
    """More gentle peak limiting to preserve speech quality"""
    try:
        audio = audio.astype(np.float64)
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        audio_norm = audio / peak
        rms = np.sqrt(np.mean(audio_norm ** 2) + 1e-10)
        current_db = 20 * np.log10(rms + 1e-10)
        gain_db = np.clip(target_db - current_db, -15, 15)  # More conservative range
        gain = 10 ** (gain_db / 20)
        audio_agc = audio_norm * gain
        return (np.tanh(audio_agc / threshold) * threshold * 32767).astype(np.float64)
    except Exception:
        return audio

def smart_vad_boost(audio, sr, boost_db=None, frame_ms=None):
    """Improved VAD with better frame handling"""
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

        # More adaptive noise floor estimation
        noise_floor = np.percentile(energies, 8)  # Lower percentile for better sensitivity
        threshold = max(noise_floor * 1.4, Config.MIN_RMS_ENERGY / 32767.0 * 0.4)
        speech_mask = energies > threshold
        speech_ratio = np.mean(speech_mask)

        # More sensitive to speech
        overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        is_soft = overall_rms < (Config.MIN_RMS_ENERGY * Config.FAR_FIELD_RMS_MULT) or speech_ratio < 0.4

        if is_soft and speech_ratio > 0.01:  # More sensitive
            # Adaptive boost based on how soft the voice is
            boost_factor = min(10 ** (boost_db / 20), 3.5)  # Higher cap
            audio_out = audio.copy()
            for i in range(n_frames):
                if speech_mask[i]:
                    audio_out[i*frame_len:(i+1)*frame_len] *= boost_factor

            # Normalize after boosting
            peak = np.max(np.abs(audio_out)) + 1e-8
            return audio_out / peak * 0.92 * 32767
        return audio
    except Exception:
        return audio

def highpass_filter(audio, sr, cutoff=60):
    """Lower cutoff to preserve more voice frequencies"""
    try:
        nyq = 0.5 * sr
        if cutoff / nyq >= 1.0:
            return audio
        b, a = signal.butter(2, cutoff / nyq, btype='high')  # 2nd order for gentler roll-off
        return signal.filtfilt(b, a, audio.astype(np.float64))
    except Exception:
        return audio

def normalize_audio(audio, target_peak=0.92):
    """More conservative normalization"""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio / peak * target_peak * 32767

def is_silent(audio, rms_threshold=0.8):
    """More sensitive silence detection"""
    if audio is None or len(audio) == 0:
        return True
    return np.sqrt(np.mean(audio.astype(np.float64)**2) + 1e-10) < rms_threshold

def estimate_snr(audio, sr):
    """More robust SNR estimation"""
    try:
        frame_len = int(sr * 0.02)
        if frame_len < 1:
            return 0.0
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return 0.0

        # Use median instead of percentile for more robustness
        energies = np.array([
            np.sqrt(np.mean(audio[i*frame_len:(i+1)*frame_len]**2) + 1e-10)
            for i in range(n_frames)
        ])

        # Sort and take middle values for better noise estimate
        sorted_energies = np.sort(energies)
        noise = np.median(sorted_energies[:len(sorted_energies)//2]) + 1e-10
        signal_p = np.median(sorted_energies[-len(sorted_energies)//4:]) + 1e-10
        return float(np.clip(20 * np.log10(signal_p / noise), 0, 80))
    except Exception:
        return 0.0

def spectral_noise_reduce(audio, sr, aggressive=False):
    """More conservative noise reduction"""
    try:
        f, t, Zxx = signal.stft(audio.astype(np.float64), fs=sr, nperseg=512)
        mag, phase = np.abs(Zxx), np.angle(Zxx)
        frame_energy = np.mean(mag**2, axis=0)

        # More conservative noise estimation
        noise_frames = frame_energy < np.percentile(frame_energy, 15)  # Higher percentile
        noise_profile = np.mean(mag[:, noise_frames], axis=1, keepdims=True) if np.sum(noise_frames) > 0 else np.min(mag, axis=1, keepdims=True)

        # Gentler reduction
        alpha = 1.8 if aggressive else 1.2
        mag_clean = np.maximum(mag - alpha * noise_profile, 0.15 * mag)  # Higher floor

        _, audio_clean = signal.istft(mag_clean * np.exp(1j * phase), fs=sr)

        if len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
        else:
            audio_clean = audio_clean[:len(audio)]

        return audio_clean
    except Exception:
        return audio

def pre_emphasis(audio, coeff=None):
    """More subtle pre-emphasis to avoid artifacts"""
    coeff = Config.PRE_EMPHASIS_COEFF if coeff is None else coeff
    try:
        audio = audio.astype(np.float64)
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    except Exception:
        return audio

def compress_dynamic_range(audio, sr, frame_ms=20, target_ratio=None, max_gain_db=None, smoothing=None):
    """Improved compressor with better parameters"""
    target_ratio = Config.COMPRESSOR_TARGET_RATIO if target_ratio is None else target_ratio
    max_gain_db = Config.COMPRESSOR_MAX_GAIN_DB if max_gain_db is None else max_gain_db
    smoothing = Config.COMPRESSOR_SMOOTHING if smoothing is None else smoothing

    try:
        audio = audio.astype(np.float64)
        frame_len = max(1, int(sr * frame_ms / 1000))
        n_frames = len(audio) // frame_len
        if n_frames < 2:
            return audio

        target_level = target_ratio * 32767
        max_gain = 10 ** (max_gain_db / 20)
        out = audio.copy()
        prev_gain = 1.0

        for i in range(n_frames):
            seg = audio[i*frame_len:(i+1)*frame_len]
            rms = np.sqrt(np.mean(seg ** 2) + 1e-8)
            if rms < 10.0:  # Very quiet segments get full boost
                gain = max_gain
            else:
                gain = min(max_gain, target_level / rms)
            gain = smoothing * prev_gain + (1 - smoothing) * gain
            prev_gain = gain
            out[i*frame_len:(i+1)*frame_len] = seg * gain

        # Final normalization
        peak = np.max(np.abs(out)) + 1e-8
        if peak > 32000:
            out = out / peak * 32000
        return out
    except Exception:
        return audio

# ============================================================
# ADAPTIVE PREPROCESSING (Optimized)
# ============================================================
def adaptive_preprocess(audio, sr):
    audio = audio.astype(np.float64)

    # Enhanced measurements
    snr = estimate_snr(audio, sr)
    overall_rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    peak = np.max(np.abs(audio))

    # More granular classification
    is_loud = peak > 24000
    is_far = overall_rms < (Config.MIN_RMS_ENERGY * Config.FAR_FIELD_RMS_MULT)
    is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 2.2)

    if is_loud:
        voice_type = "loud"
    elif is_far:
        voice_type = "far-field"
    elif is_soft:
        voice_type = "soft"
    else:
        voice_type = "normal"

    # More granular SNR zones
    if snr >= Config.SNR_SAFE_MIN:
        snr_zone = "safe"
    elif snr >= Config.SNR_DEGRADED_MIN:
        snr_zone = "degraded"
    else:
        snr_zone = "noisy"

    diagnostics = {
        "snr_db": round(snr, 1),
        "voice_type": voice_type,
        "snr_zone": snr_zone,
        "filters_applied": []
    }

    # Processing pipeline
    if snr_zone == "safe":
        audio = highpass_filter(audio, sr, 60)
        diagnostics["filters_applied"].append("highpass_60Hz")
    elif snr_zone == "degraded":
        audio = highpass_filter(audio, sr, 80)
        audio = spectral_noise_reduce(audio, sr, False)
        diagnostics["filters_applied"].append("highpass_80Hz + light_NR")
    else:  # noisy
        audio = highpass_filter(audio, sr, 100)
        audio = spectral_noise_reduce(audio, sr, True)
        diagnostics["filters_applied"].append("highpass_100Hz + NR")

    # Voice type processing
    if is_loud:
        audio = apply_agc_peak_limiter(audio, sr, target_db=-4.0)
        diagnostics["filters_applied"].append("peak_limiter")
    elif is_far:
        audio = pre_emphasis(audio)
        audio = compress_dynamic_range(audio, sr)
        diagnostics["filters_applied"].append("pre_emphasis + compression")
    elif is_soft:
        audio = smart_vad_boost(audio, sr)
        diagnostics["filters_applied"].append("smart_boost")

    # Final normalization
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
# FALLBACK MINIMAL PROCESSING
# ============================================================
def minimal_processing(audio_bytes):
    """Fallback processing when adaptive fails"""
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)

        # Very gentle processing
        audio = highpass_filter(audio, sr, 80)
        audio = normalize_audio(audio)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        wav.write(buf, sr, audio)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ============================================================
# TEXT PROCESSING (Enhanced)
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
        new_text = re.sub(
            r'\b(\d+)\s+(' + '|'.join(map(str, multipliers)) + r')\b',
            replace_match, text
        )
        if new_text == text:
            break
        text = new_text
    return text

def clean_roman_urdu(text):
    if not text:
        return text
    replacements = {
        "mujhy": "mujhe", "mujhay": "mujhe", "apky": "aapke", "apki": "aapki",
        "apko": "aapko", "kry": "kare", "krna": "karna", "krain": "karein",
        "haii": "hai", "hainn": "hain", "kia": "kya", "nhi": "nahi",
        "acha": "achha", "accha": "achha", "thik": "theek", "hain": "hain",
        "he": "hai", "ho": "hain", "ha": "ha", "ku": "kya", "ki": "ki",
        "ka": "ka", "ke": "ke", "ki": "ki", "main": "main", "mai": "main"
    }
    words = [replacements.get(w.lower(), w) for w in text.strip().split()]
    return re.sub(r'\s+', ' ', " ".join(words)).strip()

def postprocess_text(text):
    text = to_roman_urdu(text)
    text = convert_numbers_to_digits(text)
    text = combine_number_multipliers(text)
    text = clean_roman_urdu(text)
    return text

# ============================================================
# DEEPGRAM TRANSCRIPTION (Enhanced)
# ============================================================
def transcribe_deepgram(audio_bytes, last_speaker=None):
    api_key = get_api_key()
    if not api_key:
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": "DEEPGRAM_API_KEY missing"}

    params = [
        ("model", Config.DEEPGRAM_MODEL),
        ("language", Config.DEEPGRAM_LANGUAGE),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("numerals", "true"),
        ("diarize", "true"),
        ("utterances", "true"),
        ("paragraphs", "true"),  # Better for Roman Urdu
    ]

    # Add all keyterms
    for term in Config.KEYTERMS:
        params.append(("keywords", term))

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    try:
        r = requests.post(
            Config.DEEPGRAM_URL,
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=Config.DEEPGRAM_TIMEOUT
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": f"Network error: {str(e)}"}

    if r.status_code != 200:
        try:
            error_msg = r.json().get("message", r.text[:200])
        except:
            error_msg = r.text[:200]
        return {"text": "", "confidence": 0.0, "speaker_count": 0,
                "success": False, "error": f"Deepgram error: {error_msg}"}

    data = r.json()
    return process_deepgram_response(data, last_speaker)

def process_deepgram_response(data, last_speaker=None):
    try:
        channels = data.get("results", {}).get("channels", [])
        if not channels:
            return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": True}

        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": True}

        words = alternatives[0].get("words", [])
        if not words:
            text = alternatives[0].get("transcript", "").strip()
            conf = float(alternatives[0].get("confidence", 0.0) or 0.0)
            return {"text": text, "confidence": conf, "speaker_count": 1, "success": True}

        # Group by speaker
        speaker_data = {}
        for w in words:
            spk = w.get("speaker", 0)
            if spk not in speaker_data:
                speaker_data[spk] = {"words": [], "confidences": []}
            speaker_data[spk]["words"].append(w.get("word", ""))
            speaker_data[spk]["confidences"].append(float(w.get("confidence", 0.0)))

        # Select best speaker
        best_speaker = max(
            speaker_data.keys(),
            key=lambda k: len(speaker_data[k]["words"]) * np.mean(speaker_data[k]["confidences"])
        )

        # If we have a last speaker, give it preference
        if last_speaker is not None and last_speaker in speaker_data:
            best_speaker_data = speaker_data[last_speaker]
        else:
            best_speaker_data = speaker_data[best_speaker]

        text = " ".join(best_speaker_data["words"]).strip()
        conf = float(np.mean(best_speaker_data["confidences"])) if best_speaker_data["confidences"] else 0.0

        return {
            "text": text,
            "confidence": conf,
            "speaker_count": len(speaker_data),
            "success": True,
            "speaker_id": best_speaker
        }

    except Exception as e:
        try:
            alt = data["results"]["channels"][0]["alternatives"][0]
            return {
                "text": alt.get("transcript", "").strip(),
                "confidence": float(alt.get("confidence", 0.0) or 0.0),
                "speaker_count": 1,
                "success": True
            }
        except:
            return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": True}

# ============================================================
# MAIN PROCESSING PIPELINE (with fallback)
# ============================================================
def process_voice_input(audio_bytes, last_speaker=None):
    # First try with full processing
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0,
                "speaker_count": 0, "diagnostics": None, "error": "Audio too quiet or too short"}

    result = transcribe_deepgram(cleaned["bytes"], last_speaker)

    if result["success"] and result["text"] and result["text"] != "[No clear speech detected]":
        return {
            "success": True,
            "text": postprocess_text(result["text"]),
            "confidence": result["confidence"],
            "speaker_count": result["speaker_count"],
            "diagnostics": cleaned["diagnostics"],
            "speaker_id": result.get("speaker_id"),
            "error": None
        }

    # If first attempt failed, try with minimal processing
    minimal_bytes = minimal_processing(audio_bytes)
    if minimal_bytes:
        retry_result = transcribe_deepgram(minimal_bytes, last_speaker)
        if retry_result["success"] and retry_result["text"] and retry_result["text"] != "[No clear speech detected]":
            cleaned["diagnostics"]["filters_applied"].append("retry_with_minimal_processing")
            return {
                "success": True,
                "text": postprocess_text(retry_result["text"]),
                "confidence": retry_result["confidence"],
                "speaker_count": retry_result["speaker_count"],
                "diagnostics": cleaned["diagnostics"],
                "speaker_id": retry_result.get("speaker_id"),
                "error": None
            }

    # If both attempts failed
    return {
        "success": True,
        "text": "[No clear speech detected]",
        "confidence": 0.0,
        "speaker_count": result.get("speaker_count", 0),
        "diagnostics": cleaned["diagnostics"],
        "speaker_id": result.get("speaker_id"),
        "error": None
    }

# ============================================================
# FEEDBACK SYSTEM
# ============================================================
def get_feedback(diagnostics):
    if not diagnostics:
        return None

    vt = diagnostics.get("voice_type")
    zone = diagnostics.get("snr_zone")

    if vt == "loud":
        return "🔊 Came through loud and clear — I normalized it for best results."
    if vt == "far-field":
        return "📏 Detected distant voice — I boosted and compressed it for clarity."
    if vt == "soft":
        return "🔈 Quiet voice detected — I amplified it. Try speaking a bit louder next time."
    if zone == "noisy":
        return "🌪️ Noisy environment detected — I cleaned it up, but a quieter space would help."
    if zone == "degraded":
        return "🌤️ Some background noise — I reduced it while preserving your voice."
    return "✅ Clear signal — no adjustments needed."

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎤 Voice Agent Pro")
st.caption("Optimized for clear transcription | Enhanced Roman Urdu support | Better distance handling")

# Initialize session state
defaults = {
    "last_text": "",
    "last_conf": 0.0,
    "last_speakers": 0,
    "last_diag": None,
    "last_speaker_id": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Check for API key
if not get_api_key():
    st.error("""
    DEEPGRAM_API_KEY missing.
    Add it to your `.env` file (local) or Streamlit Secrets (deployed):
    DEEPGRAM_API_KEY=your_key_here
    """)
    st.stop()

# Recording UI
st.subheader("Record Your Voice")
audio_output = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="🛑 Stop Recording",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="mic"
)

if audio_output and audio_output.get("bytes"):
    with st.spinner("Processing audio..."):
        result = process_voice_input(audio_output["bytes"], st.session_state.last_speaker_id)

    if result["success"]:
        st.session_state.last_text = result["text"]
        st.session_state.last_conf = result["confidence"]
        st.session_state.last_speakers = result["speaker_count"]
        st.session_state.last_diag = result.get("diagnostics")
        if result.get("speaker_id") is not None:
            st.session_state.last_speaker_id = result["speaker_id"]
        st.success("Transcription complete!")
    else:
        st.session_state.last_diag = result.get("diagnostics")
        st.error(result.get("error", "Processing failed"))

# Feedback
feedback = get_feedback(st.session_state.last_diag)
if feedback:
    st.info(feedback)

# Diagnostics
if st.session_state.last_diag:
    d = st.session_state.last_diag
    with st.expander("🔬 Audio Diagnostics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SNR", f"{d.get('snr_db', 0):.1f} dB")
        with col2:
            st.metric("Voice Type", d.get("voice_type", "?"))
        with col3:
            st.metric("Noise Level", d.get("snr_zone", "?").title())

        st.write(f"**Filters Applied:** `{', '.join(d.get('filters_applied', [])) or 'none'}`")

st.divider()
st.subheader("📝 Transcription")

if st.session_state.last_text:
    st.markdown(
        f"""
        <div style="background:#1e1e2e;padding:20px;border-radius:12px;
                    color:#cdd6f4;font-size:18px;line-height:1.65;">
            {st.session_state.last_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Confidence: {st.session_state.last_conf:.2f} | "
        f"Speakers: {st.session_state.last_speakers}"
    )
else:
    st.info("Record something to see transcription here")

if st.button("🗑️ Clear", use_container_width=True):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()
