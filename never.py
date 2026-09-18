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
from datetime import datetime, timezone

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

    # --- Distance / far-field compensation (NEW) ---
    # A recording is treated as "far / weak" when its overall level sits
    # below this multiple of MIN_RMS_ENERGY, even if the room is quiet.
    FAR_FIELD_RMS_MULT   = 3.2
    COMPRESSOR_TARGET_RATIO = 0.34   # fraction of int16 range each speech frame is pulled toward
    COMPRESSOR_MAX_GAIN_DB  = 20.0   # ceiling so we don't just amplify hiss
    COMPRESSOR_SMOOTHING    = 0.35   # 0-1, higher = smoother/slower gain changes (less "pumping")
    PRE_EMPHASIS_COEFF      = 0.30   # mild high-frequency lift (consonants decay fastest with distance)

    # --- Voice profile / lock (NEW — replaces the old Deepgram-speaker-id lock) ---
    PROFILE_DIR = Path("voice_profiles")
    ENROLL_MIN_SEC = 2.5
    ENROLL_MAX_SEC = 8.0
    ENROLL_RECOMMENDED_CLIPS = 3
    MATCH_SIMILARITY_THRESHOLD = 0.70   # a speaker cluster must reach this to be "you"
    MATCH_MARGIN = 0.07                 # ...and beat the next-closest speaker by this much
    MAX_CLUSTER_SECONDS_FOR_EMBEDDING = 6.0

Config.PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# API KEY (Streamlit Cloud + Local compatible)
# ============================================================
def get_api_key():
    try:
        key = st.secrets.get("DEEPGRAM_API_KEY")
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("DEEPGRAM_API_KEY")
    if key:
        return key
    return None

# ============================================================
# AUDIO UTILS (existing)
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
# DISTANCE / FAR-FIELD COMPENSATION (NEW)
# ============================================================
def pre_emphasis(audio, coeff=None):
    """
    Mild high-frequency lift. Consonants (which carry most speech
    intelligibility) are higher-frequency and lose energy fastest over
    distance/room absorption, so a gentle lift here helps far-field
    speech sound more "present" without just turning up everything
    (which would also raise the noise floor).
    """
    coeff = Config.PRE_EMPHASIS_COEFF if coeff is None else coeff
    try:
        audio = audio.astype(np.float64)
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    except Exception:
        return audio

def compress_dynamic_range(audio, sr, frame_ms=25, target_ratio=None, max_gain_db=None, smoothing=None):
    """
    Frame-wise compressor: instead of one flat gain for the whole clip
    (which either under-boosts quiet parts or clips loud parts), each
    frame's gain is computed from ITS OWN level and smoothed across time
    to avoid audible "pumping". This is what actually helps a speaker who
    is far from the mic — a distant voice has both quiet AND
    louder-but-still-far moments, and a flat multiplier handles neither
    well.
    """
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
            if rms < 12.0:
                # near-silence — don't amplify noise/hiss in the gaps
                gain = 1.0
            else:
                gain = float(np.clip(target_level / rms, 1.0, max_gain))
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
# ADAPTIVE PIPELINE (UPDATED: now branches into far-field compensation)
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
    elif is_far:
        # NEW: distance compensation chain — gentle high-frequency lift,
        # then a real per-frame compressor instead of one flat boost.
        audio = pre_emphasis(audio)
        audio = compress_dynamic_range(audio, sr)
        diagnostics["filters_applied"].append("pre_emphasis + dynamic_compression (far-field)")
    elif is_soft:
        audio = smart_vad_boost(audio, sr)
        diagnostics["filters_applied"].append("soft_boost")

    audio = normalize_audio(audio)
    return np.clip(audio, -32768, 32767).astype(np.int16), diagnostics

def preprocess_audio(audio_bytes):
    """Returns cleaned wav bytes AND the raw int16 array + sample rate,
    because the voice-lock matching step (below) needs to crop the exact
    audio that was sent to Deepgram."""
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
        return {
            "bytes": buf.read(),
            "duration": duration,
            "diagnostics": diagnostics,
            "audio_int16": audio_clean,
            "sample_rate": int(sr),
        }
    except Exception:
        return None

# ============================================================
# VOICE PROFILE / LOCK — real voiceprint matching (NEW)
# ============================================================
# This replaces the old "Lock Current Speaker" button, which locked onto
# Deepgram's speaker_id (0, 1, 2...). That ID is only meaningful WITHIN one
# API call — a fresh recording gets a fresh diarization, so "speaker 0"
# today has no guaranteed relation to "speaker 0" tomorrow. That's why the
# old lock felt unreliable. This version enrolls an actual voiceprint
# (via Resemblyzer) once, and matches every future recording against it —
# so the lock survives across separate recordings/sessions.

_voice_encoder = None

def get_voice_encoder():
    global _voice_encoder
    if _voice_encoder is not None:
        return _voice_encoder
    from resemblyzer import VoiceEncoder
    _voice_encoder = VoiceEncoder()
    return _voice_encoder

def _int16_to_float(audio_int16):
    return audio_int16.astype(np.float32) / 32768.0

def extract_embedding(audio_int16, sr):
    """Returns an L2-normalized voiceprint, or None if the clip is too
    short / has no detectable voice activity."""
    try:
        from resemblyzer import preprocess_wav
        encoder = get_voice_encoder()
        wav_float = _int16_to_float(np.asarray(audio_int16, dtype=np.float64))
        processed = preprocess_wav(wav_float, source_sr=sr)
        if len(processed) < int(sr * 0.3):
            return None
        emb = encoder.embed_utterance(processed)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-9 else emb
    except Exception as e:
        print(f"[warn] embedding extraction failed: {e}")
        return None

def extract_embedding_averaged(clips):
    """clips: list of (audio_int16, sr). Averaging 2-3 clips recorded at
    different moments makes the profile noticeably more stable than one
    single clip, and cuts down false-matches between similar-sounding
    voices."""
    embs = [e for e in (extract_embedding(a, sr) for a, sr in clips) if e is not None]
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else mean

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _safe_name(name):
    safe = "".join(c for c in name if c.isalnum() or c in "-_").strip()
    return safe or f"user_{int(datetime.now().timestamp())}"

def save_profile(name, embedding, metadata=None):
    safe = _safe_name(name)
    emb_path = Config.PROFILE_DIR / f"{safe}.npy"
    meta_path = Config.PROFILE_DIR / f"{safe}.json"
    np.save(emb_path, embedding)
    meta = dict(metadata or {})
    meta.update({
        "display_name": name,
        "safe_name": safe,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def load_profile(name):
    safe = _safe_name(name)
    emb_path = Config.PROFILE_DIR / f"{safe}.npy"
    meta_path = Config.PROFILE_DIR / f"{safe}.json"
    if not emb_path.exists():
        return None, None
    embedding = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return embedding, meta

def list_profiles():
    profiles = []
    for meta_path in Config.PROFILE_DIR.glob("*.json"):
        try:
            profiles.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(profiles, key=lambda m: m.get("created_at", ""), reverse=True)

def delete_profile(name):
    safe = _safe_name(name)
    for ext in (".npy", ".json"):
        p = Config.PROFILE_DIR / f"{safe}{ext}"
        if p.exists():
            p.unlink()

def enrollment_quality_check(audio_int16, sr):
    """Lightweight quality gate for enrollment clips, reusing the SNR/
    silence helpers already in this file."""
    duration = len(audio_int16) / sr
    if duration < Config.ENROLL_MIN_SEC:
        return False, f"Too short ({duration:.1f}s, need ≥ {Config.ENROLL_MIN_SEC}s)."
    if is_silent(audio_int16.astype(np.float64)):
        return False, "Too quiet — move closer to the mic and try again."
    snr = estimate_snr(audio_int16.astype(np.float64), sr)
    if snr < Config.SNR_DEGRADED_MIN:
        return False, f"Too much background noise (SNR {snr:.1f} dB). Record somewhere quieter."
    return True, f"OK — SNR {snr:.1f} dB, {duration:.1f}s."

# ============================================================
# SPEAKER GROUPING (from Deepgram diarization words)
# ============================================================
def group_words_by_speaker(words, gap_tolerance=0.6):
    """Turns Deepgram's flat word list into per-local-speaker text +
    merged time segments (so we can crop the matching audio later)."""
    by_speaker = {}
    for w in words:
        spk = w.get("speaker", 0)
        by_speaker.setdefault(spk, []).append(w)

    groups = {}
    for spk, ws in by_speaker.items():
        text = " ".join(w.get("word", "") for w in ws).strip()
        confs = [float(w.get("confidence", 0.0)) for w in ws]
        conf = float(np.mean(confs)) if confs else 0.0

        segments = []
        cur_start, cur_end = ws[0].get("start", 0.0), ws[0].get("end", 0.0)
        for w in ws[1:]:
            s, e = w.get("start", cur_end), w.get("end", cur_end)
            if s - cur_end <= gap_tolerance:
                cur_end = e
            else:
                segments.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        segments.append((cur_start, cur_end))

        groups[spk] = {
            "text": text,
            "confidence": conf,
            "segments": segments,
            "word_count": len(ws),
            "talk_time": sum(e - s for s, e in segments),
        }
    return groups

def crop_concat(audio_int16, sr, segments, max_sec=None):
    pieces, total = [], 0.0
    for s, e in segments:
        i0, i1 = int(s * sr), min(int(e * sr), len(audio_int16))
        if i1 <= i0:
            continue
        pieces.append(audio_int16[i0:i1])
        total += (i1 - i0) / sr
        if max_sec and total >= max_sec:
            break
    if not pieces:
        return np.array([], dtype=audio_int16.dtype)
    return np.concatenate(pieces)

def find_target_speaker(cluster_embeddings, target_embedding):
    """cluster_embeddings: {local_speaker_id: embedding_or_None}.
    Returns (matched_id_or_None, debug_dict)."""
    scored = {spk: cosine_similarity(emb, target_embedding)
              for spk, emb in cluster_embeddings.items() if emb is not None}
    if not scored:
        return None, {"scores": {}, "reason": "no usable speaker audio in this recording"}

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    best_spk, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    debug = {"scores": {k: round(v, 3) for k, v in scored.items()},
              "best": best_spk, "best_score": round(best_score, 3),
              "second_score": round(second_score, 3)}

    if best_score < Config.MATCH_SIMILARITY_THRESHOLD:
        debug["reason"] = f"best match only {best_score:.2f} (threshold {Config.MATCH_SIMILARITY_THRESHOLD})"
        return None, debug
    if (best_score - second_score) < Config.MATCH_MARGIN:
        debug["reason"] = (f"best ({best_score:.2f}) too close to next speaker "
                            f"({second_score:.2f}) — likely similar-sounding voices")
        return None, debug

    debug["reason"] = "matched"
    return best_spk, debug

# ============================================================
# LEGACY AUTO-PICK (kept for "Auto" mode / no enrollment yet)
# ============================================================
def pick_better_speaker(deepgram_data, last_speaker=None):
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

def postprocess_text(text):
    return clean_roman_urdu(combine_number_multipliers(convert_numbers_to_digits(to_roman_urdu(text))))

# ============================================================
# DEEPGRAM (now returns raw words too, for voiceprint matching)
# ============================================================
def call_deepgram(audio_bytes):
    api_key = get_api_key()
    if not api_key:
        return None, "DEEPGRAM_API_KEY missing"
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
        return None, f"Network: {e}"
    if r.status_code != 200:
        return None, f"Deepgram {r.status_code}"
    return r.json(), None

def extract_words(deepgram_data):
    try:
        return deepgram_data["results"]["channels"][0]["alternatives"][0].get("words", [])
    except Exception:
        return []

# ============================================================
# MAIN PROCESS (now mode-aware: "auto" vs "locked")
# ============================================================
def process_voice_input(audio_bytes, mode="auto", last_speaker=None, profile_name=None):
    cleaned = preprocess_audio(audio_bytes)
    if cleaned is None:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": None, "speaker_id": None, "error": "Audio too quiet or too short.",
                "match_debug": None}

    data, err = call_deepgram(cleaned["bytes"])
    if err:
        return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                "diagnostics": cleaned["diagnostics"], "speaker_id": None, "error": err,
                "match_debug": None}

    words = extract_words(data)
    groups = group_words_by_speaker(words) if words else {}
    speaker_count = len(groups) if groups else 1

    if mode == "locked" and profile_name:
        target_embedding, meta = load_profile(profile_name)
        if target_embedding is None:
            return {"success": False, "text": "", "confidence": 0.0, "speaker_count": speaker_count,
                    "diagnostics": cleaned["diagnostics"], "speaker_id": None,
                    "error": f"No enrolled profile named '{profile_name}'. Enroll first.",
                    "match_debug": None}

        if not groups:
            return {"success": False, "text": "", "confidence": 0.0, "speaker_count": 0,
                    "diagnostics": cleaned["diagnostics"], "speaker_id": None,
                    "error": "No speech detected in this recording.", "match_debug": None}

        cluster_embs = {}
        for spk, g in groups.items():
            seg_audio = crop_concat(cleaned["audio_int16"], cleaned["sample_rate"], g["segments"],
                                     max_sec=Config.MAX_CLUSTER_SECONDS_FOR_EMBEDDING)
            cluster_embs[spk] = extract_embedding(seg_audio, cleaned["sample_rate"]) if len(seg_audio) else None

        matched_spk, debug = find_target_speaker(cluster_embs, target_embedding)
        if matched_spk is None:
            return {"success": False, "text": "", "confidence": 0.0, "speaker_count": speaker_count,
                    "diagnostics": cleaned["diagnostics"], "speaker_id": None,
                    "error": "Couldn't confidently find your enrolled voice in this recording "
                             f"({debug.get('reason')}). Try Auto mode, move closer to the mic, "
                             "or re-enroll in a quieter spot.",
                    "match_debug": debug}

        text = postprocess_text(groups[matched_spk]["text"])
        conf = groups[matched_spk]["confidence"]
        return {"success": True, "text": text or "[No clear speech detected]", "confidence": conf,
                "speaker_count": speaker_count, "diagnostics": cleaned["diagnostics"],
                "speaker_id": matched_spk, "error": None, "match_debug": debug}

    # ---- Auto mode (no profile / not enrolled yet) ----
    text, conf, spk_count, speaker_id = pick_better_speaker(data, last_speaker=last_speaker)
    text = postprocess_text(text)
    return {"success": True, "text": text or "[No clear speech detected]", "confidence": conf,
            "speaker_count": spk_count, "diagnostics": cleaned["diagnostics"],
            "speaker_id": speaker_id, "error": None, "match_debug": None}

# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🎤 Voice Agent")
st.caption("Real voiceprint lock (works across recordings) + far-field listening boost")

defaults = {
    "last_text": "", "last_conf": 0.0, "last_speakers": 0,
    "last_diag": None, "last_speaker_id": None,
    "enroll_clips": [], "match_debug": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not get_api_key():
    st.error("DEEPGRAM_API_KEY missing. Add it to your `.env` file:\n\nDEEPGRAM_API_KEY=your_key_here")
    st.stop()

tab_use, tab_enroll = st.tabs(["🎙️ Use", "🔒 Enroll My Voice"])

# ------------------------------------------------------------
# ENROLL TAB
# ------------------------------------------------------------
with tab_enroll:
    st.subheader("Enroll your voice (once)")
    st.info(
        f"Record {Config.ENROLL_RECOMMENDED_CLIPS} short clips ({Config.ENROLL_MIN_SEC:.0f}-"
        f"{Config.ENROLL_MAX_SEC:.0f}s each) of just your own voice, in a quiet spot. "
        "Recording more than one clip makes the lock far more reliable — this is what actually "
        "fixes the old 'lock' button, which only remembered a speaker NUMBER that changed every "
        "recording, not your actual voice."
    )
    profile_name = st.text_input("Profile name (e.g. your name)", key="profile_name_input")

    rec = mic_recorder(start_prompt="🎙️ Record enrollment clip", stop_prompt="🛑 Stop",
                        just_once=True, format="wav", key="enroll_mic")
    if rec and rec.get("bytes"):
        sr, raw = wav.read(io.BytesIO(rec["bytes"]))
        if len(raw.shape) > 1:
            raw = raw.mean(axis=1)
        raw = raw.astype(np.int16)
        ok, msg = enrollment_quality_check(raw, sr)
        if ok:
            st.success(f"✅ {msg}")
            st.session_state.enroll_clips.append((raw, sr))
        else:
            st.error(f"❌ {msg}")

    st.write(f"Clips collected: **{len(st.session_state.enroll_clips)}** "
             f"(recommended: {Config.ENROLL_RECOMMENDED_CLIPS})")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save profile", disabled=not st.session_state.enroll_clips or not profile_name):
            embedding = extract_embedding_averaged(st.session_state.enroll_clips)
            if embedding is None:
                st.error("Couldn't build a voiceprint from those clips — try recording again, closer to the mic.")
            else:
                save_profile(profile_name, embedding, metadata={"num_clips": len(st.session_state.enroll_clips)})
                st.success(f"Voice profile '{profile_name}' saved.")
                st.session_state.enroll_clips = []
    with col2:
        if st.button("🗑️ Clear collected clips"):
            st.session_state.enroll_clips = []
            st.rerun()

    st.divider()
    st.write("**Saved profiles**")
    profiles = list_profiles()
    if not profiles:
        st.caption("None yet.")
    for p in profiles:
        c1, c2 = st.columns([4, 1])
        c1.write(f"👤 {p.get('display_name')} — {p.get('created_at', '')[:19]}")
        if c2.button("Delete", key=f"del_{p.get('safe_name')}"):
            delete_profile(p.get("display_name"))
            st.rerun()

# ------------------------------------------------------------
# USE TAB
# ------------------------------------------------------------
with tab_use:
    profiles = list_profiles()
    profile_names = [p["display_name"] for p in profiles]

    mode_label = st.radio(
        "Mode",
        ["🎯 Auto (best guess, no enrollment needed)", "🔒 My Voice (enrolled profile)"],
        horizontal=False,
    )
    mode = "locked" if mode_label.startswith("🔒") else "auto"

    selected_profile = None
    if mode == "locked":
        if not profile_names:
            st.warning("No enrolled profile yet — go to the 'Enroll My Voice' tab first.")
        else:
            selected_profile = st.selectbox("Use profile", profile_names)

    audio_output = mic_recorder(
        start_prompt="Start Recording", stop_prompt="Stop Recording",
        just_once=True, use_container_width=True, format="wav", key="mic",
    )

    if audio_output and audio_output.get("bytes"):
        with st.spinner("Processing..."):
            result = process_voice_input(
                audio_output["bytes"], mode=mode,
                last_speaker=st.session_state.last_speaker_id,
                profile_name=selected_profile,
            )
        if result["success"]:
            st.session_state.last_text = result["text"]
            st.session_state.last_conf = result["confidence"]
            st.session_state.last_speakers = result["speaker_count"]
            st.session_state.last_diag = result.get("diagnostics")
            st.session_state.match_debug = result.get("match_debug")
            if result.get("speaker_id") is not None:
                st.session_state.last_speaker_id = result["speaker_id"]
            st.success("Done")
        else:
            st.session_state.last_diag = result.get("diagnostics")
            st.session_state.match_debug = result.get("match_debug")
            st.error(result.get("error", "Failed"))

    if st.session_state.last_diag:
        d = st.session_state.last_diag
        with st.expander("Diagnostics"):
            st.write(f"SNR: **{d.get('snr_db')} dB** | Voice: **{d.get('voice_type')}** | Zone: **{d.get('snr_zone')}**")
            st.write(f"Filters: `{', '.join(d.get('filters_applied', []))}`")
            if st.session_state.match_debug:
                st.write("Voice match scores:")
                st.json(st.session_state.match_debug)

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
        for k in ("last_text", "last_conf", "last_speakers", "last_diag", "match_debug"):
            st.session_state[k] = defaults[k]
        st.rerun()
