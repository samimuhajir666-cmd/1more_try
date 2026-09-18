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

    # Voice / noise thresholds
    SNR_SAFE_MIN         = 28.0
    SNR_DEGRADED_MIN     = 16.0
    MIN_RMS_ENERGY       = 22.0
    SOFT_BOOST_DB        = 11.0
    VAD_FRAME_MS         = 25
    HIGHPASS_HZ          = 80
    MIN_DURATION_SECONDS = 0.4
    MAX_DURATION_SECONDS = 120

    # Distance / far-field compensation
    FAR_FIELD_RMS_MULT      = 3.2
    COMPRESSOR_TARGET_RATIO = 0.22   # gentler than before — avoids over-driving quiet clips into artifacts
    COMPRESSOR_MAX_GAIN_DB  = 12.0   # was 20 — 20dB combined with pre-emphasis was distorting speech
    COMPRESSOR_SMOOTHING    = 0.35
    PRE_EMPHASIS_COEFF      = 0.15   # was 0.30 — a lighter lift is less likely to amplify noise into garbage

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
# CORE AUDIO UTILS
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

# ============================================================
# DISTANCE / FAR-FIELD COMPENSATION
# ============================================================
def pre_emphasis(audio, coeff=None):
    """Mild high-frequency lift — consonants (the part that actually
    carries meaning) are the first thing to fade with distance."""
    coeff = Config.PRE_EMPHASIS_COEFF if coeff is None else coeff
    try:
        audio = audio.astype(np.float64)
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    except Exception:
        return audio

def compress_dynamic_range(audio, sr, frame_ms=25, target_ratio=None, max_gain_db=None, smoothing=None):
    """Per-frame compressor instead of one flat gain — a real human ear
    (and this) adapts moment-to-moment to a distant/quiet voice rather
    than applying a single volume knob to the whole clip."""
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
            gain = 1.0 if rms < 12.0 else float(np.clip(target_level / rms, 1.0, max_gain))
            gain = smoothing * prev_gain + (1 - smoothing) * gain
            prev_gain = gain
            out[i*frame_len:(i+1)*frame_len] = seg * gain
        peak = np.max(np.abs(out)) + 1e-8
        if peak > 32000:
            out = out / peak * 32000
        return out
    except Exception:
        return audio

# ============================================================
# ADAPTIVE PIPELINE
# ============================================================
def adaptive_preprocess(audio, sr):
    audio = audio.astype(np.float64)
    snr = estimate_snr(audio, sr)
    overall_rms = np.sqrt(np.mean(audio**2) + 1e-10)
    peak = np.max(np.abs(audio))
    is_loud = peak > 25000
    is_far = overall_rms < (Config.MIN_RMS_ENERGY * Config.FAR_FIELD_RMS_MULT)
    is_soft = overall_rms < (Config.MIN_RMS_ENERGY * 2.5)
    voice_type = "loud" if is_loud else ("far/soft" if is_far else ("soft" if is_soft else "normal"))
    snr_zone = "safe" if snr >= Config.SNR_SAFE_MIN else ("degraded" if snr >= Config.SNR_DEGRADED_MIN else "critical")

    diagnostics = {"snr_db": round(snr, 1), "voice_type": voice_type, "snr_zone": snr_zone, "filters_applied": []}

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
    elif is_far:
        audio = pre_emphasis(audio)
        audio = compress_dynamic_range(audio, sr)
        diagnostics["filters_applied"].append("pre_emphasis + dynamic_compression (far-field)")
    elif is_soft:
        audio = smart_vad_boost(audio, sr)
        diagnostics["filters_applied"].append("soft_boost")

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
# SPEAKER SELECTION (auto — picks whoever is clearest/most consistent,
# no enrollment, no lock, no profile)
# ============================================================
def pick_better_speaker(deepgram_data, last_speaker=None):
    try:
        channels = deepgram_data.get("results", {}).get("channels", [])
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
                score *= 1.4
            scores[spk] = score
        if not scores:
            return "", 0.0, 0
        best = max(scores, key=scores.get)
        data = speaker_data[best]
        text = " ".join(data["words"]).strip()
        conf = float(np.mean(data["confidences"])) if data["confidences"] else 0.0
        return text, conf, len(speaker_data)
    except Exception:
        try:
            alt = deepgram_data["results"]["channels"][0]["alternatives"][0]
            return alt.get("transcript", "").strip(), float(alt.get("confidence", 0.0)), 1
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

def postprocess_text(text):
    return clean_roman_urdu(combine_number_multipliers(convert_numbers_to_digits(to_roman_urdu(text))))

# ============================================================
# DEEPGRAM
# ============================================================
def transcribe_deepgram(audio_bytes, last_speaker=None):
    api_key = get_api_key()
    if not api_key:
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False, "error": "DEEPGRAM_API_KEY missing"}
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
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False, "error": f"Network: {e}"}
    if r.status_code != 200:
        # Show Deepgram's actual error body instead of just the status code —
        # this is almost always a key/plan/param problem, and the real
        # message tells you exactly which one.
        return {"text": "", "confidence": 0.0, "speaker_count": 0, "success": False,
                "error": f"Deepgram {r.status_code}: {r.text[:500]}"}
    data = r.json()
    text, conf, spk_count = pick_better_speaker(data, last_speaker=last_speaker)
    text = postprocess_text(text)
    if not text or len(text.strip()) < 2:
        return {"text": "[No clear speech detected]", "confidence": 0.0, "speaker_count": spk_count, "success": True, "error": None, "raw_word_count": len(data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("words", []))}
    return {"text": text, "confidence": conf, "speaker_count": spk_count, "success": True, "error": None}

# ============================================================
# MAIN PROCESS (with automatic safe-retry if heavy processing
# accidentally garbles a clean recording)
# ============================================================
def _minimal_clean(audio_bytes):
    """A deliberately light touch: just high-pass filter + normalize,
    no gain-boost / compression / pre-emphasis. Used as a fallback when
    the full adaptive chain returns nothing — protects against the case
    where aggressive far-field processing distorts an already-clear clip."""
    try:
        sr, audio = wav.read(io.BytesIO(audio_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float64)
        audio = highpass_filter(audio, sr, 80)
        audio = normalize_audio(audio)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        wav.write(buf, sr, audio)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def process_voice_input(audio_bytes, last_speaker=None):
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": None, "error": "Audio too quiet or too short."}

    result = transcribe_deepgram(cleaned["bytes"], last_speaker=last_speaker)
    if not result["success"]:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": cleaned["diagnostics"], "error": result.get("error")}

    if "raw_word_count" in result:
        cleaned["diagnostics"]["raw_word_count"] = result["raw_word_count"]

    # Safety net: if the fully-processed audio produced nothing, the
    # processing itself may have distorted a perfectly fine recording —
    # retry once with minimal, non-destructive cleaning.
    if result["text"] == "[No clear speech detected]":
        light_bytes = _minimal_clean(audio_bytes)
        if light_bytes:
            retry = transcribe_deepgram(light_bytes, last_speaker=last_speaker)
            if retry["success"] and retry["text"] != "[No clear speech detected]":
                cleaned["diagnostics"]["filters_applied"].append("retried_with_minimal_processing")
                return {"success": True, "text": retry["text"], "confidence": retry["confidence"],
                        "speaker_count": retry["speaker_count"], "diagnostics": cleaned["diagnostics"], "error": None}
            elif "raw_word_count" in retry:
                cleaned["diagnostics"]["raw_word_count_retry"] = retry["raw_word_count"]

    return {"success": True, "text": result["text"], "confidence": result["confidence"],
            "speaker_count": result["speaker_count"], "diagnostics": cleaned["diagnostics"], "error": None}

# ============================================================
# HUMAN-LIKE LISTENING FEEDBACK
# ============================================================
def listening_feedback(diagnostics):
    """Plain-language read on how well the mic actually heard you —
    the kind of thing a person would tell you, not raw numbers."""
    if not diagnostics:
        return None
    vt, zone = diagnostics.get("voice_type"), diagnostics.get("snr_zone")
    if vt == "loud":
        return "🔊 Came through loud — I evened it out automatically."
    if vt == "far/soft":
        return "📏 Sounded far from the mic — I boosted it, but standing a bit closer will help clarity."
    if vt == "soft":
        return "🔈 A little quiet — I boosted it. Speak a touch louder for best accuracy."
    if zone == "critical":
        return "🌪️ Noisy background — I cleaned it up as much as possible, but a quieter spot will help."
    if zone == "degraded":
        return "🌤️ Some background noise — cleaned up, should still be accurate."
    return "✅ Clear signal — nothing needed fixing."

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎤 Voice Agent")
st.caption("Just record — no setup, no profiles. Tuned to hear you clearly even from a distance or in noise.")

defaults = {"last_text": "", "last_conf": 0.0, "last_speakers": 0, "last_diag": None, "last_speaker_id": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not get_api_key():
    st.error("DEEPGRAM_API_KEY missing. Add it to `.env` (local) or Streamlit **Settings → Secrets** (deployed):\n\nDEEPGRAM_API_KEY=your_key_here")
    st.stop()

audio_output = mic_recorder(
    start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop Recording",
    just_once=True, use_container_width=True, format="wav", key="mic",
)

if audio_output and audio_output.get("bytes"):
    with st.spinner("Listening..."):
        result = process_voice_input(audio_output["bytes"])
    if result["success"]:
        st.session_state.last_text = result["text"]
        st.session_state.last_conf = result["confidence"]
        st.session_state.last_speakers = result["speaker_count"]
        st.session_state.last_diag = result.get("diagnostics")
        st.success("Done")
    else:
        st.session_state.last_diag = result.get("diagnostics")
        st.error(result.get("error", "Failed"))

feedback = listening_feedback(st.session_state.last_diag)
if feedback:
    st.info(feedback)

if st.session_state.last_diag:
    d = st.session_state.last_diag
    with st.expander("Technical details"):
        st.write(f"SNR: **{d.get('snr_db')} dB** | Voice: **{d.get('voice_type')}** | Zone: **{d.get('snr_zone')}**")
        st.write(f"Filters applied: `{', '.join(d.get('filters_applied', []))}`")
        if "raw_word_count" in d:
            st.write(f"Words Deepgram actually returned: **{d.get('raw_word_count')}**")
        if "raw_word_count_retry" in d:
            st.write(f"Words on retry (minimal processing): **{d.get('raw_word_count_retry')}**")

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
    st.info("Record something to see transcription here.")

if st.button("Clear", use_container_width=True):
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()
