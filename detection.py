"""Shared, local inference for the web app and evaluation CLI.

Scores are uncalibrated model outputs, not probabilities of authenticity.
No downloaded repository code is executed. Model versions and labels are pinned.
"""
#detection.py
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import threading

cache_path = str((Path(__file__).parent / ".cache" / "huggingface").resolve())
if os.name == "nt":
    # Deep project paths otherwise exceed Windows' legacy 260-character limit.
    cache_path = "\\\\?\\" + cache_path
os.environ.setdefault("HF_HOME", cache_path)
# The project environment is prone to failing on the experimental Xet transfer.
# Disabling it avoids the intermittent download corruption and DNS-related failures.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
from PIL import Image, ImageOps

MODELS = {
    "image": ("aaronkantrowitz/ai-human-generated-image-detection", "67eacdf4e1953353f8d2c0ca22fde0f3226f1049", {"human": 0, "AI-generated": 1}),
    "text": ("Hello-SimpleAI/chatgpt-detector-roberta", "d2b342c61775d5dd0221808a79983ed3b86ffd86", {"Human": 0, "ChatGPT": 1}),
    "audio": ("MelodyMachine/Deepfake-audio-detection", "8fa126974c64f5ba601484de0446ddf1839f58f9", {"fake": 1, "real": 0}),
}
LOCK = threading.RLock()


@dataclass(frozen=True)
class Settings:
    frame_count: int = 32
    audio_seconds: int = 6
    max_audio_segments: int = 40
    max_audio_duration: int = 600
    max_text_chunks: int = 32
    min_text_words: int = 80
    real_threshold: float = 0.20
    fake_threshold: float = 0.80
    image_real_threshold: float = 0.0
    disagreement_range: float = 0.60


def pipeline_id(settings):
    payload = {"version": 1, "models": MODELS, "settings": asdict(settings)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fake_score(predictions, modality):
    """Fail closed on unknown labels, missing classes, NaN or invalid scores."""
    mapping = MODELS[modality][2]
    scores = {}
    for prediction in predictions:
        label, score = prediction["label"], float(prediction["score"])
        if label not in mapping or label in scores or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid {modality} model prediction: {label}")
        scores[label] = score
    if set(scores) != set(mapping) or not math.isclose(sum(scores.values()), 1, abs_tol=0.01):
        raise ValueError(f"Incomplete {modality} model scores")
    return sum(scores[label] for label, is_fake in mapping.items() if is_fake)


@lru_cache(maxsize=3)
def get_pipeline(modality):
    import torch
    from transformers import AutoImageProcessor, pipeline
    model, revision, labels = MODELS[modality]
    task = {"image": "image-classification", "text": "text-classification", "audio": "audio-classification"}[modality]
    # Text publisher supplies only .bin weights. torch >=2.6 loads them with
    # weights_only protection; other models use safetensors exclusively.
    extra = {}
    if modality == "image":
        extra["image_processor"] = AutoImageProcessor.from_pretrained(
            model, revision=revision, use_fast=False, trust_remote_code=False)
    classifier = pipeline(task, model=model, revision=revision,
                          device=0 if torch.cuda.is_available() else -1,
                          trust_remote_code=False,
                          model_kwargs={"use_safetensors": modality != "text"}, **extra)
    if set(classifier.model.config.id2label.values()) != set(labels):
        raise ValueError(f"Unexpected label mapping for {model}")
    classifier.model.eval()
    return classifier


def sample_indices(total, count):
    if total <= 0 or count <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(total, count), dtype=int).tolist()))


class Detector:
    def __init__(self, settings=None, calibration_path=None):
        self.settings = settings or Settings()
        self.thresholds = {}
        if calibration_path:
            config = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            if config.get("pipeline_id") != pipeline_id(self.settings):
                raise ValueError("Threshold file does not match this model/pipeline version")
            self.thresholds = config["thresholds"]
            for limits in self.thresholds.values():
                if not 0 <= limits["real"] < limits["fake"] <= 1:
                    raise ValueError("Invalid decision thresholds")

    def result(self, modality, scores, warnings=None, force_uncertain=False, **details):
        warnings = list(warnings or [])
        limits = self.thresholds.get(modality, {})
        default_low = self.settings.image_real_threshold if modality in {"image", "video"} else self.settings.real_threshold
        low = limits.get("real", default_low)
        high = limits.get("fake", self.settings.fake_threshold)
        score = float(np.mean(scores)) if scores else None
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid aggregate score")
        if scores and max(scores) - min(scores) >= self.settings.disagreement_range:
            warnings.append("Different parts of the content give conflicting signals.")
            force_uncertain = True
        status = "uncertain"
        if score is not None and not force_uncertain:
            status = "fake" if score >= high else "real" if score <= low else "uncertain"
        if status == "uncertain" and not warnings:
            warnings.append("The evidence is insufficient for a reliable classification.")
        return {
            "status": status,
            "label": {"fake": "Likely synthetic / manipulated", "real": "Likely authentic", "uncertain": "Inconclusive"}[status],
            "is_threat": None if status == "uncertain" else status == "fake",
            # Raw classifier scores are not calibrated probabilities.
            "confidence": None,
            "fake_score": score, "calibrated": False,
            "threshold_source": "validation" if limits else "conservative defaults (unvalidated)",
            "warnings": warnings, "samples_analyzed": len(scores),
            "sample_scores": [float(s) for s in scores],
            "pipeline_id": pipeline_id(self.settings), **details,
        }

    def image_score(self, image):
        with LOCK:
            return fake_score(get_pipeline("image")(image, top_k=None), "image")

    def image(self, data):
        with Image.open(io.BytesIO(data)) as source:
            animated = getattr(source, "n_frames", 1) > 1
            image = ImageOps.exif_transpose(source).convert("RGB")
        warnings = []
        if min(image.size) < 96:
            warnings.append("Image resolution is too low for a dependable classification.")
        if animated:
            warnings.append("Only the first frame of this animated image was analyzed; upload a video for temporal sampling.")
        score = self.image_score(image)
        return self.result("image", [score], warnings, force_uncertain=bool(warnings),
                           models=[MODELS["image"][0]])

    def text(self, text):
        if len(text.split()) < self.settings.min_text_words:
            return self.result("text", [], [f"Provide at least {self.settings.min_text_words} words; short text is unreliable."])
        # The selected detector was trained for English human/ChatGPT writing.
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isascii() for c in letters) / len(letters) < 0.8:
            return self.result("text", [], ["This detector is intended for English text; this input is outside its supported scope."])
        if len(text) > 500_000:
            raise ValueError("Text exceeds the 500,000-character processing limit")
        with LOCK:
            classifier = get_pipeline("text")
            tokenizer = classifier.tokenizer
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            window = min(tokenizer.model_max_length, 512) - tokenizer.num_special_tokens_to_add(pair=False)
            chunks = [ids[i:i+window] for i in range(0, len(ids), window)]
            selected = sample_indices(len(chunks), self.settings.max_text_chunks)
            scores, weights = [], []
            import torch
            for index in selected:
                chunk_ids = chunks[index]
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                batch = tokenizer(
                    chunk_text,
                    return_tensors="pt",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    padding=True
                    )
                batch = {key: value.to(classifier.device) for key, value in batch.items()}
                with torch.inference_mode():
                    probabilities = classifier.model(**batch).logits.softmax(-1)[0].cpu().tolist()
                predictions = [{"label": classifier.model.config.id2label[i], "score": p} for i, p in enumerate(probabilities)]
                scores.append(fake_score(predictions, "text"))
                weights.append(len(chunks[index]))
        partial = len(selected) < len(chunks)
        warnings = ["AI-writing detection does not establish authorship or whether statements are true."]
        if partial:
            warnings.append("Text exceeds the processing budget; only distributed chunks were checked.")
        result = self.result("text", scores, warnings, partial, tokens_analyzed=sum(weights),
                             total_tokens=len(ids), coverage=sum(weights)/len(ids), models=[MODELS["text"][0]])
        # Weight by token count so a short final chunk cannot dominate the text.
        weighted = self.result("text", [float(np.average(scores, weights=weights))], warnings,
                               partial or (max(scores)-min(scores) >= self.settings.disagreement_range))
        for key in ("status", "label", "is_threat", "confidence", "fake_score"):
            result[key] = weighted[key]
        return result

    def audio(self, path):
        import imageio_ffmpeg
        import soundfile as sf
        # Decode a bounded duration in a subprocess. This supports audio/video
        # containers without loading an arbitrarily long recording into RAM.
        with tempfile.TemporaryDirectory() as temp:
            decoded = Path(temp) / "audio.wav"
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-y", "-i", str(path),
                       "-vn", "-t", str(self.settings.max_audio_duration+1), "-ac", "1", "-ar", "16000", str(decoded)]
            process = subprocess.run(command, capture_output=True, timeout=120)
            if process.returncode:
                raise ValueError("Audio track could not be decoded or is missing")
            samples, rate = sf.read(decoded, dtype="float32")
        duration = len(samples)/rate
        if duration < 1:
            return self.result("audio", [], ["At least one second of audible speech is required."])
        if duration > self.settings.max_audio_duration:
            return self.result("audio", [], [f"Audio exceeds the {self.settings.max_audio_duration}-second limit; split it into shorter clips."])
        window = self.settings.audio_seconds * rate
        segments = [samples[i:i+window] for i in range(0, len(samples), window)]
        selected = sample_indices(len(segments), self.settings.max_audio_segments)
        scores, spans, skipped = [], [], 0
        for index in selected:
            segment = segments[index]
            if len(segment) < rate // 2 or float(np.sqrt(np.mean(segment**2))) < 1e-4:
                skipped += 1
                continue
            with LOCK:
                prediction = get_pipeline("audio")({"array": segment, "sampling_rate": rate}, top_k=None)
            scores.append(fake_score(prediction, "audio"))
            spans.append([index*self.settings.audio_seconds, min((index+1)*self.settings.audio_seconds, duration)])
        partial = len(selected) < len(segments)
        warnings = ["Audio detection is intended for speech; music and other sounds are outside its validated scope."]
        if partial:
            warnings.append("Audio was sampled across its duration; unsampled sections remain unchecked.")
        if skipped:
            warnings.append(f"Skipped {skipped} silent or very short segments.")
        if not scores:
            warnings.append("No usable speech-length audio segments were found.")
        return self.result("audio", scores, warnings, partial or not scores,
                           duration_seconds=duration, coverage=sum(b-a for a,b in spans)/duration,
                           segments=spans, models=[MODELS["audio"][0]])

    def video(self, path):
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if total <= 0:
                raise ValueError("Video contains no readable frames")
            selected = sample_indices(total, self.settings.frame_count)
            scores, times, low_resolution = [], [], False
            for index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = capture.read()
                if not success:
                    continue
                low_resolution = low_resolution or min(frame.shape[:2]) < 96
                scores.append(self.image_score(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
                times.append(index/fps if fps > 0 else None)
        finally:
            capture.release()
        if not scores:
            raise ValueError("Video frames could not be analyzed")
        warnings = ["Sampled frames cannot rule out brief edits or establish lip-sync authenticity."]
        incomplete = len(scores) != len(selected)
        if incomplete:
            warnings.append("Some selected frames could not be read.")
        if low_resolution:
            warnings.append("Video resolution is too low for a dependable classification.")
        visual = self.result("video", scores, warnings, incomplete or low_resolution,
                             frames_analyzed=len(scores), total_frames=total, timestamps=times)
        try:
            audio = self.audio(path)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            audio = self.result("audio", [], [f"Audio not assessed: {exc}"])
        # Preserve separate evidence. An unvalidated mean/fusion of two different
        # modalities is not a calibrated probability and can hide contradictions.
        if (audio["status"] == "fake" and visual["status"] != "fake") or (audio["status"] == "uncertain" and audio["fake_score"] is not None):
            visual.update(status="uncertain", label="Inconclusive", is_threat=None)
            visual["warnings"].append("Audio and visual evidence require separate review; no validated fusion model is installed.")
        visual["warnings"].extend(audio["warnings"])
        if visual["status"] == "real":
            visual["label"] = "Sampled frames likely authentic"
        visual["audio"] = audio
        visual["models"] = [MODELS["image"][0], MODELS["audio"][0]]
        return visual

    def file(self, modality, path):
        path = Path(path)
        if modality == "image":
            return self.image(path.read_bytes())
        if modality == "text":
            return self.text(path.read_text(encoding="utf-8"))
        if modality in {"audio", "video"}:
            return getattr(self, modality)(path)
        raise ValueError(f"Unsupported modality: {modality}")


@lru_cache(maxsize=1)
def default_detector():
    return Detector(calibration_path=os.environ.get("DETECTION_THRESHOLDS"))


def _reality_defender_scan(path, api_key):
    import asyncio
    from realitydefender import RealityDefender

    async def scan():
        client = RealityDefender(api_key=api_key)
        try:
            upload = await client.upload(file_path=str(path))
            result = await client.get_result(upload["request_id"])
            result["request_id"] = upload["request_id"]
            return result
        finally:
            await client.cleanup()

    def run_scan():
        return asyncio.run(scan())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_scan()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(run_scan).result()


def _reality_defender_cached(file_hash, result=None):
    from app import app as flask_app, db
    from sqlalchemy import text

    with flask_app.app_context():
        engine = db.engine
        if engine.dialect.name != "sqlite":
            raise RuntimeError("Detection cache requires SQLite")
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE IF NOT EXISTS reality_defender_cache ("
                "file_sha256 TEXT PRIMARY KEY, p_fake REAL NOT NULL, raw_json TEXT NOT NULL)"
            ))
            if result is None:
                row = connection.execute(
                    text("SELECT p_fake, raw_json FROM reality_defender_cache WHERE file_sha256 = :file_hash"),
                    {"file_hash": file_hash},
                ).mappings().first()
                if row:
                    import json
                    return {"p_fake": row["p_fake"], "raw": json.loads(row["raw_json"])}
                return None

            import json
            connection.execute(
                text("INSERT OR IGNORE INTO reality_defender_cache (file_sha256, p_fake, raw_json) "
                     "VALUES (:file_hash, :p_fake, :raw_json)"),
                {"file_hash": file_hash, "p_fake": result["p_fake"],
                 "raw_json": json.dumps(result["raw"], default=str)},
            )


def _detect_with_reality_defender(path):
    import hashlib
    from dotenv import load_dotenv

    try:
        load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
        api_key = os.environ.get("REALITY_DEFENDER_API_KEY")
        if not api_key:
            return {"p_fake": None, "error": "Missing REALITY_DEFENDER_API_KEY"}

        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hash = digest.hexdigest()

        with LOCK:
            cached = _reality_defender_cached(file_hash)
            if cached is not None:
                return cached

            # This is the only place to swap in a different model later.
            raw = _reality_defender_scan(path, api_key)
            if raw.get("score") is None:
                return {
                    "p_fake": None,
                    "error": "no score returned",
                    "status": raw.get("status"),
                    "raw": raw,
                }
            p_fake = float(raw["score"])
            if not 0 <= p_fake <= 1:
                raise ValueError("Detection service returned an invalid score")
            result = {"p_fake": p_fake, "raw": raw}
            _reality_defender_cached(file_hash, result)
            return result
    except Exception as error:
        message = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
        return {"p_fake": None, "error": message[:160]}


def detect_image(path):
    return _detect_with_reality_defender(path)


def detect_audio(path):
    return _detect_with_reality_defender(path)


MIN_AUDIO_DURATION = 1.0


def detect_video(path):
    import cv2
    import imageio_ffmpeg
    import shutil
    import wave

    capture = cv2.VideoCapture(str(path))
    temp_dir = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError("Video has no readable frames or valid frame rate")

        duration = frame_count / fps
        timestamps = [duration / 2] if duration < 1 else [
            timestamp for timestamp in (0.5, 1.5, 2.5) if timestamp < duration
        ]
        samples = []
        seen_indices = set()
        for timestamp in timestamps:
            frame_index = min(int(timestamp * fps), frame_count - 1)
            if frame_index not in seen_indices:
                samples.append((timestamp, frame_index))
                seen_indices.add(frame_index)

        temp_dir = tempfile.mkdtemp()
        frame_scores = []
        analyzed_timestamps = []
        frame_errors = []
        metadata_hint = []
        for sample_number, (timestamp, frame_index) in enumerate(samples):
            analyzed_timestamps.append(timestamp)
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    raise ValueError(f"could not read video frame {frame_index}")

                frame_path = os.path.join(temp_dir, f"frame_{sample_number}.jpg")
                if not cv2.imwrite(
                    frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise ValueError(f"could not save temporary frame {frame_index}")
                frame_result = detect_image(frame_path)
                frame_scores.append(frame_result.get("p_fake"))
                frame_error = frame_result.get("error")
                if frame_error or frame_result.get("p_fake") is None:
                    frame_errors.append({
                        "timestamp": timestamp,
                        "error": frame_error or "no score returned",
                    })

                raw = frame_result.get("raw")
                hint = frame_result.get("metadata_hint")
                if hint is None and isinstance(raw, dict):
                    hint = raw.get("metadata_hint", raw.get("metadata"))
                if hint is not None:
                    metadata_hint.append({"timestamp": timestamp, "metadata": hint})
            except Exception as error:
                frame_scores.append(None)
                frame_errors.append({"timestamp": timestamp, "error": str(error)})

        audio_path = os.path.join(temp_dir, "audio.wav")
        audio_score = None
        audio_error = None
        try:
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", audio_path,
            ]
            extraction = subprocess.run(command, capture_output=True, timeout=120)
            error_text = extraction.stderr.decode("utf-8", errors="replace")
            lower_error = error_text.casefold()
            no_audio_stream = any(phrase in lower_error for phrase in (
                "matches no streams", "does not contain any stream",
                "stream specifier", "could not find output stream",
            ))
            if no_audio_stream:
                audio_error = "no audio track"
            elif extraction.returncode != 0:
                audio_error = error_text[-300:] or "audio extraction failed"
            elif not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
                audio_error = error_text[-300:] or "audio extraction produced no file"
            else:
                with wave.open(audio_path, "rb") as audio_file:
                    audio_duration = audio_file.getnframes() / audio_file.getframerate()
                if audio_duration < MIN_AUDIO_DURATION:
                    audio_error = "audio shorter than minimum duration"
                else:
                    audio_result = detect_audio(audio_path)
                    audio_score = audio_result.get("p_fake")
                    if audio_result.get("error") or audio_score is None:
                        audio_error = audio_result.get("error") or "no score returned"
                    raw = audio_result.get("raw")
                    hint = audio_result.get("metadata_hint")
                    if hint is None and isinstance(raw, dict):
                        hint = raw.get("metadata_hint", raw.get("metadata"))
                    if hint is not None:
                        metadata_hint.append({"modality": "audio", "metadata": hint})
        except Exception as error:
            audio_error = str(error)

        valid_frame_scores = [score for score in frame_scores if score is not None]
        max_frame = max(valid_frame_scores) if valid_frame_scores else None
        fake_frame_count = sum(
            score is not None and score >= 0.5 for score in frame_scores
        )

        if audio_score is None or any(score is None for score in frame_scores):
            verdict = "inconclusive"
        else:
            frame_is_fake = max_frame >= 0.5
            audio_is_fake = audio_score >= 0.5
            if frame_is_fake != audio_is_fake:
                verdict = "conflict"
            elif frame_is_fake or audio_is_fake:
                verdict = "fake"
            else:
                verdict = "real"

        return {
            "timestamps": analyzed_timestamps,
            "frame_scores": frame_scores,
            "max_frame": max_frame,
            "fake_frame_count": fake_frame_count,
            "audio": audio_score,
            "frame_errors": frame_errors,
            "audio_error": audio_error,
            "verdict": verdict,
            **({"metadata_hint": metadata_hint} if metadata_hint else {}),
        }
    finally:
        capture.release()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
