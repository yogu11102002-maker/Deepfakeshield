import os
os.environ["PATH"] += r";C:\Users\yogu1\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin"
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
import os
import time
import tempfile
import requests
from transformers import pipeline
from dotenv import load_dotenv
import io
import csv
from PIL import Image
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# Fusion model is optional — the app works fine without it (falls back to
# frame-only voting for videos). It only gets used once you've run
# train_fusion_model.py and a fusion_model.pkl file exists.
try:
    import joblib
    FUSION_MODEL = joblib.load("fusion_model.pkl") if os.path.exists("fusion_model.pkl") else None
except Exception:
    FUSION_MODEL = None

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit (videos are bigger)

# Image upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)


# ---------- HUGGING FACE CONFIG ----------

HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')
HF_API_URL = "https://router.huggingface.co/hf-inference/models"

IMAGE_MODEL = "dima806/deepfake_vs_real_image_detection"
TEXT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"

# ---------- AUDIO MODEL (local, runs on GPU if available) ----------

AUDIO_DEVICE = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
AUDIO_TARGET_SR = 16000          # model's native sample rate — resampling to this avoids
                                  # pipeline doing it internally on every call
AUDIO_MAX_SECONDS = 15           # deepfake artifacts are audible within a few seconds,
                                  # so we don't need to process the whole clip

AUDIO_MODEL = pipeline(
    "audio-classification",
    model="MelodyMachine/Deepfake-audio-detection",
    device=AUDIO_DEVICE,
)


def _warmup_audio_model():
    """First call to a HF pipeline lazily finishes loading weights onto the
    device, so it's much slower than every call after it. Run one dummy
    inference at startup so that hit happens now, not on a user's request."""
    try:
        silence = np.zeros(AUDIO_TARGET_SR, dtype="float32")
        AUDIO_MODEL(silence)
        print(f"[audio model] warmed up on {'GPU' if AUDIO_DEVICE == 0 else 'CPU'}")
    except Exception as exc:
        print(f"[audio model] warmup skipped: {exc}")


_warmup_audio_model()


def _prep_audio_for_model(path, max_seconds=AUDIO_MAX_SECONDS):
    """Trims to the first `max_seconds` and resamples to the model's native
    sample rate before inference. Falls back to the original file untouched
    if librosa/soundfile aren't installed.

    Returns (path_to_use, cleanup_path_or_None).
    """
    if not HAS_LIBROSA:
        return path, None

    try:
        y, sr = librosa.load(path, sr=AUDIO_TARGET_SR, mono=True, duration=max_seconds)
        trimmed_path = path + "_trim.wav"
        sf.write(trimmed_path, y, sr)
        return trimmed_path, trimmed_path
    except Exception:
        # If trimming fails for any reason, just fall back to the original file
        return path, None


def classify_audio_bytes_local(audio_path):
    """Runs the local audio pipeline on a file path and returns 'fake'/'real'."""
    prepped_path, cleanup_path = _prep_audio_for_model(audio_path)
    try:
        result = AUDIO_MODEL(prepped_path)
        top = max(result, key=lambda r: r["score"])
        return "fake" if "fake" in top["label"].lower() else "real"
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except OSError:
                pass


VIDEO_FRAMES_TO_SAMPLE = 10   # frames pulled evenly across the video and run through IMAGE_MODEL


# ---------- DATABASE MODELS ----------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    content_type = db.Column(db.String(20), nullable=False)   # 'image' or 'text'
    source_name = db.Column(db.String(255))                   # filename or text snippet
    image_path = db.Column(db.String(255))                    # path to saved image file
    label = db.Column(db.String(80))
    confidence = db.Column(db.Float)
    is_threat = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------- AUTH DECORATOR ----------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ---------- FILE HELPERS ----------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload_file(file_storage, user_id):
    """Save uploaded image to disk and return relative path"""
    if not file_storage or not allowed_file(file_storage.filename):
        return None

    filename = secure_filename(file_storage.filename)
    timestamp = int(time.time())
    filename = f"{user_id}_{timestamp}_{filename}"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)

    return f"/uploads/{filename}"


def create_thumbnail(image_path, size=(150, 150)):
    """Create thumbnail for analysis history"""
    try:
        full_path = os.path.join(os.path.dirname(__file__), image_path.lstrip('/'))
        if os.path.exists(full_path):
            img = Image.open(full_path)
            img.thumbnail(size)
            return True
    except:
        pass
    return False


# ---------- HUGGING FACE HELPERS ----------

def _hf_headers(content_type=None):
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def query_hf_model(model_name, data=None, json_payload=None, content_type=None, retries=3):
    url = f"{HF_API_URL}/{model_name}"
    headers = _hf_headers(content_type)

    for attempt in range(retries):
        try:
            if json_payload is not None:
                response = requests.post(url, headers=headers, json=json_payload, timeout=30)
            else:
                response = requests.post(url, headers=headers, data=data, timeout=30)
        except requests.exceptions.RequestException as exc:
            return None, f"Could not reach Hugging Face API: {exc}"

        if response.status_code == 200:
            return response.json(), None

        if response.status_code == 503:
            try:
                wait_time = min(response.json().get("estimated_time", 5), 15)
            except Exception:
                wait_time = 5
            time.sleep(wait_time)
            continue

        if response.status_code == 401:
            return None, "Hugging Face token missing or invalid. Set the HF_API_TOKEN environment variable."

        return None, f"Hugging Face API error ({response.status_code}): {response.text[:200]}"

    return None, "The model is still warming up on Hugging Face. Please try again in a few seconds."


def flatten_predictions(result):
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]

    if not isinstance(result, list):
        return []

    predictions = [p for p in result if isinstance(p, dict) and "label" in p and "score" in p]
    predictions.sort(key=lambda p: p["score"], reverse=True)
    return predictions


def classify_image_bytes(image_bytes, mimetype="image/jpeg"):
    """Low-level helper: sends raw image bytes to IMAGE_MODEL and returns
    (label, confidence) for the top prediction. Used for both direct image
    uploads and individual video frames."""
    result, error = query_hf_model(IMAGE_MODEL, data=image_bytes, content_type=mimetype)
    if error:
        return None, None, error

    predictions = flatten_predictions(result)
    if not predictions:
        return None, None, "Unexpected response from the image model."

    top = predictions[0]
    return top["label"], round(top["score"] * 100, 2), None


def analyze_image(file_storage):
    file_bytes = file_storage.read()
    mimetype = file_storage.mimetype or "application/octet-stream"

    label, confidence, error = classify_image_bytes(file_bytes, mimetype)
    if error:
        return None, error

    is_threat = "fake" in label.lower() or "deepfake" in label.lower()
    return {"label": label, "confidence": confidence, "is_threat": is_threat}, None


def _extract_and_score_audio_track(video_path):
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        return None

    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_tmp_path = tmp.name

        # Only pull out the first AUDIO_MAX_SECONDS of audio instead of the
        # whole track — much faster for long videos and enough signal for
        # the deepfake-audio model to make a call.
        end_time = min(AUDIO_MAX_SECONDS, clip.duration) if clip.duration else AUDIO_MAX_SECONDS
        clip.audio.subclip(0, end_time).write_audiofile(
            audio_tmp_path, fps=AUDIO_TARGET_SR, logger=None
        )
        clip.close()

        try:
            result = AUDIO_MODEL(audio_tmp_path)          # local pipeline call, no HF API
            top = max(result, key=lambda r: r["score"])
            is_fake = "fake" in top["label"].lower()
            return 1.0 if is_fake else 0.0
        finally:
            os.remove(audio_tmp_path)

    except Exception:
        return None


def analyze_video(file_storage, num_frames=VIDEO_FRAMES_TO_SAMPLE):
    """Samples a handful of evenly-spaced frames from the video, runs each
    one through the image deepfake model, and aggregates the results by
    majority vote (ties broken toward 'fake', the safer default)."""

    suffix = os.path.splitext(file_storage.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    try:
        capture = cv2.VideoCapture(tmp_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            capture.release()
            return None, "Could not read any frames from this video."

        sample_count = min(num_frames, total_frames)
        frame_indices = [
            int(total_frames * i / sample_count) for i in range(sample_count)
        ]

        frame_results = []
        for idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = capture.read()
            if not success:
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            label, confidence, error = classify_image_bytes(buffer.tobytes(), "image/jpeg")
            if error:
                continue

            frame_results.append((label, confidence))

        capture.release()

        if not frame_results:
            return None, "Could not analyze any frames from this video (Hugging Face API issue)."

        fake_votes = [c for l, c in frame_results if "fake" in l.lower() or "deepfake" in l.lower()]
        real_votes = [c for l, c in frame_results if not ("fake" in l.lower() or "deepfake" in l.lower())]
        video_fake_fraction = len(fake_votes) / len(frame_results)

        # ----- Fusion path: combine the video-frame score with the video's
        # own audio-track score using the trained fusion model, if we have one -----
        if FUSION_MODEL is not None:
            audio_fake_score = _extract_and_score_audio_track(tmp_path)
            if audio_fake_score is not None:
                probability_fake = FUSION_MODEL.predict_proba([[video_fake_fraction, audio_fake_score]])[0][1]
                is_threat = probability_fake >= 0.5
                confidence = round((probability_fake if is_threat else 1 - probability_fake) * 100, 2)
                label = "Deepfake (fusion)" if is_threat else "Real (fusion)"

                return {
                    "label": label,
                    "confidence": confidence,
                    "is_threat": is_threat,
                    "frames_analyzed": len(frame_results),
                }, None

        # ----- Fallback: frame-only majority vote (used when no fusion
        # model is trained yet, or audio extraction failed for this file) -----
        is_threat = len(fake_votes) >= len(real_votes)
        winning_votes = fake_votes if is_threat else real_votes
        avg_confidence = round(sum(winning_votes) / len(winning_votes), 2)
        label = "Deepfake" if is_threat else "Real"

        return {
            "label": label,
            "confidence": avg_confidence,
            "is_threat": is_threat,
            "frames_analyzed": len(frame_results),
        }, None

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def analyze_audio(file_storage):
    suffix = os.path.splitext(file_storage.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
    # ↑ 'with' block yahin khatam ho gaya, tempfile ka apna handle band ho gaya

    file_storage.save(tmp_path)   # ← ab clean, fresh handle se save ho rha hai

    prepped_path, cleanup_path = _prep_audio_for_model(tmp_path)

    try:
        result = AUDIO_MODEL(prepped_path)
        top = max(result, key=lambda r: r["score"])
        label = top["label"]
        confidence = round(top["score"] * 100, 2)
        is_threat = any(k in label.lower() for k in ["fake", "spoof", "synthetic"])
        return {"label": label, "confidence": confidence, "is_threat": is_threat}, None
    except Exception as exc:
        return None, f"Audio analysis failed: {exc}"
    finally:
        if cleanup_path:
            try:
                os.remove(cleanup_path)
            except OSError:
                pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def analyze_text(text):
    result, error = query_hf_model(TEXT_MODEL, json_payload={"inputs": text})
    if error:
        return None, error

    predictions = flatten_predictions(result)
    if not predictions:
        return None, "Unexpected response from the text model."

    top = predictions[0]
    label = top["label"]
    confidence = round(top["score"] * 100, 2)
    is_threat = "fake" in label.lower() or "ai" in label.lower()

    return {"label": label, "confidence": confidence, "is_threat": is_threat}, None


# ---------- PAGE ROUTES ----------

@app.route("/")
def home():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return render_template("home.html", user=user)


@app.route("/login")
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("login.html")


@app.route("/signup")
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("signup.html")


@app.route("/dashboard")
@login_required
def dashboard():
    user = User.query.get(session['user_id'])

    analyses = (
        Analysis.query
        .filter_by(user_id=user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    total = len(analyses)
    threats = sum(1 for a in analyses if a.is_threat)
    safe = total - threats
    protection_score = 100 if total == 0 else round((safe / total) * 100)

    recent = analyses[:5]

    return render_template(
        "dashboard.html",
        user=user,
        total=total,
        safe=safe,
        threats=threats,
        protection_score=protection_score,
        recent=recent,
    )


@app.route("/analyze")
@login_required
def analyze():
    user = User.query.get(session['user_id'])
    return render_template("analyze.html", user=user)


@app.route("/history")
@login_required
def history():
    user = User.query.get(session['user_id'])

    content_type = request.args.get('type', 'all')
    threat_filter = request.args.get('threat', 'all')

    query = Analysis.query.filter_by(user_id=user.id)

    if content_type != 'all':
        query = query.filter_by(content_type=content_type)

    if threat_filter == 'threat':
        query = query.filter_by(is_threat=True)
    elif threat_filter == 'safe':
        query = query.filter_by(is_threat=False)

    analyses = query.order_by(Analysis.created_at.desc()).all()

    return render_template("history.html", user=user, analyses=analyses)


@app.route("/settings")
@login_required
def settings():
    user = User.query.get(session['user_id'])
    return render_template("settings.html", user=user)


@app.route("/reports")
@login_required
def reports():
    user = User.query.get(session['user_id'])

    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()

    total = len(analyses)
    threats = sum(1 for a in analyses if a.is_threat)
    safe = total - threats

    image_count = sum(1 for a in analyses if a.content_type == 'image')
    text_count = sum(1 for a in analyses if a.content_type == 'text')

    avg_confidence = round(sum(a.confidence for a in analyses) / total, 2) if total > 0 else 0

    return render_template(
        "reports.html",
        user=user,
        total=total,
        threats=threats,
        safe=safe,
        image_count=image_count,
        text_count=text_count,
        avg_confidence=avg_confidence,
        analyses=analyses
    )


@app.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename):
    """Serve uploaded images (only to logged-in users)"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


# ---------- API ROUTES ----------

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    confirm_password = data.get("confirmPassword") or ""

    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match."}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "An account with this email already exists."}), 409

    user = User(name=name, email=email)
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    return jsonify({"success": True, "message": "Account created successfully!"}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    session['user_id'] = user.id
    session['user_name'] = user.name

    return jsonify({"success": True, "message": "Login successful!"}), 200


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True}), 200


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    uploaded_file = request.files.get("file")
    text_input = (request.form.get("text") or "").strip()

    if not uploaded_file and not text_input:
        return jsonify({"success": False, "message": "Upload a file or enter text to analyze."}), 400

    image_path = None

    # ----- IMAGE -----
    if uploaded_file and uploaded_file.mimetype.startswith("image/"):
        result, error = analyze_image(uploaded_file)
        content_type = "image"
        source_name = uploaded_file.filename

        # Reset the stream position — analyze_image() already read it to EOF
        # to send to Hugging Face, so we must rewind before saving to disk.
        uploaded_file.stream.seek(0)

        # Save image to disk
        image_path = save_upload_file(uploaded_file, session['user_id'])

    # ----- TEXT -----
    elif text_input:
        result, error = analyze_text(text_input)
        content_type = "text"
        source_name = text_input[:80]

    # ----- VIDEO -----
    elif uploaded_file and uploaded_file.mimetype.startswith("video/"):
        result, error = analyze_video(uploaded_file)
        content_type = "video"
        source_name = uploaded_file.filename

    # ----- AUDIO -----
    elif uploaded_file and uploaded_file.mimetype.startswith("audio/"):
        result, error = analyze_audio(uploaded_file)
        content_type = "audio"
        source_name = uploaded_file.filename

    # ----- UNSUPPORTED FILE TYPE -----
    elif uploaded_file:
        return jsonify({
            "success": False,
            "message": "Unsupported file type. Please upload an image, video, audio file, or enter text."
        }), 400

    else:
        return jsonify({"success": False, "message": "Nothing to analyze."}), 400

    if error:
        return jsonify({"success": False, "message": error}), 502

    analysis = Analysis(
        user_id=session['user_id'],
        content_type=content_type,
        source_name=source_name,
        image_path=image_path,
        label=result["label"],
        confidence=result["confidence"],
        is_threat=result["is_threat"],
    )
    db.session.add(analysis)
    db.session.commit()

    return jsonify({
        "success": True,
        "content_type": content_type,
        "label": result["label"],
        "confidence": result["confidence"],
        "is_threat": result["is_threat"],
        "frames_analyzed": result.get("frames_analyzed"),
    }), 200


@app.route("/api/change-password", methods=["POST"])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}

    user = User.query.get(session['user_id'])
    old_password = data.get("oldPassword") or ""
    new_password = data.get("newPassword") or ""
    confirm_password = data.get("confirmPassword") or ""

    if not user.check_password(old_password):
        return jsonify({"success": False, "message": "Current password is incorrect."}), 401

    if new_password != confirm_password:
        return jsonify({"success": False, "message": "New passwords do not match."}), 400

    if len(new_password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400

    user.set_password(new_password)
    db.session.commit()

    return jsonify({"success": True, "message": "Password changed successfully!"}), 200


@app.route("/api/update-profile", methods=["POST"])
@login_required
def api_update_profile():
    data = request.get_json(silent=True) or {}

    user = User.query.get(session['user_id'])
    name = (data.get("name") or "").strip()

    if not name:
        return jsonify({"success": False, "message": "Name cannot be empty."}), 400

    user.name = name
    db.session.commit()
    session['user_name'] = name

    return jsonify({"success": True, "message": "Profile updated!"}), 200


@app.route("/api/delete-account", methods=["POST"])
@login_required
def api_delete_account():
    user = User.query.get(session['user_id'])

    # Delete analyses
    Analysis.query.filter_by(user_id=user.id).delete()

    # Delete user
    db.session.delete(user)
    db.session.commit()

    session.clear()

    return jsonify({"success": True, "message": "Account deleted."}), 200


@app.route("/api/download-csv")
@login_required
def download_csv():
    user = User.query.get(session['user_id'])
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Type', 'Source', 'Result', 'Confidence %', 'Status'])

    for analysis in analyses:
        writer.writerow([
            analysis.created_at.strftime('%Y-%m-%d %H:%M'),
            analysis.content_type.capitalize(),
            analysis.source_name[:50],
            analysis.label,
            analysis.confidence,
            'Threat' if analysis.is_threat else 'Safe'
        ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'deepfake-shield-report-{datetime.now().strftime("%Y%m%d")}.csv'
    )


# ---------- DB INIT ----------

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)