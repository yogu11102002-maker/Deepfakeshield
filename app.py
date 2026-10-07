import os
import json
import logging
import base64
import hashlib
import smtplib
import urllib.error
import urllib.request
import secrets
import math
from email.message import EmailMessage
from urllib.parse import urljoin
from authlib.integrations.flask_client import OAuth
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime, timezone
import time
import tempfile
from dotenv import load_dotenv
import io
import csv
from PIL import Image
from detection import default_detector, detect_audio as detect_audio_reality_defender
from detection import detect_image as detect_image_reality_defender
from detection import detect_video
from sqlalchemy import inspect, text as sql_text
from sqlalchemy.exc import IntegrityError

load_dotenv()

REALITY_DEFENDER_FAKE_THRESHOLD = 0.5
REALITY_DEFENDER_REAL_THRESHOLD = 0.3

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB upload limit (videos are bigger)
app.config['GOOGLE_CLIENT_ID'] = os.environ.get('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.environ.get('GOOGLE_CLIENT_SECRET')
app.config['GOOGLE_REDIRECT_URI'] = os.environ.get(
    'GOOGLE_REDIRECT_URI', 'http://localhost:5000/login/google/callback'
)
app.config['PASSWORD_RESET_BASE_URL'] = os.environ.get(
    'PASSWORD_RESET_BASE_URL'
) or os.environ.get(
    'RENDER_EXTERNAL_URL', 'http://127.0.0.1:5000'
)
app.config['SMTP_HOST'] = os.environ.get('SMTP_HOST', '')
app.config['SMTP_PORT'] = os.environ.get('SMTP_PORT', '587')
app.config['SMTP_USE_TLS'] = os.environ.get('SMTP_USE_TLS', '1') == '1'
app.config['SMTP_USE_SSL'] = os.environ.get('SMTP_USE_SSL', '0') == '1'
app.config['SMTP_USERNAME'] = os.environ.get('SMTP_USERNAME', '')
app.config['SMTP_PASSWORD'] = os.environ.get('SMTP_PASSWORD', '')
app.config['SMTP_FROM'] = os.environ.get('SMTP_FROM', '')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# Image upload folder
UPLOAD_FOLDER = os.environ.get(
    'UPLOAD_FOLDER', os.path.join(os.path.dirname(__file__), 'uploads')
)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)


# ---------- DATABASE MODELS ----------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    google_sub = db.Column(db.String(255), nullable=True)
    email_verified = db.Column(db.Boolean, nullable=False, default=False)
    email_verification_token_hash = db.Column(db.String(64), nullable=True)
    email_verification_expires_at = db.Column(db.Integer, nullable=True)
    password_reset_token_hash = db.Column(db.String(64), nullable=True)
    password_reset_expires_at = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        self.password_reset_token_hash = None
        self.password_reset_expires_at = None

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
    is_threat = db.Column(db.Boolean, nullable=True)
    evidence_json = db.Column(db.Text, nullable=True)

    @property
    def status(self):
        if self.content_type == 'video':
            details = self.video_details
            if details.get('analysis_error'):
                return 'failed'
            verdict = details.get('verdict')
            return verdict if verdict in {'fake', 'real', 'conflict', 'inconclusive'} else 'inconclusive'
        return 'uncertain' if self.is_threat is None else 'fake' if self.is_threat else 'real'

    @property
    def video_details(self):
        if self.content_type != 'video' or not self.evidence_json:
            return {}
        try:
            details = json.loads(self.evidence_json)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(details, dict):
            return {}

        details = details.copy()
        audio_value = details.get('audio')
        if isinstance(audio_value, dict):
            if not details.get('audio_error'):
                details['audio_error'] = audio_value.get('error') or audio_value.get('message')
            audio_value = next((audio_value.get(key) for key in ('p_fake', 'fake_score', 'score')
                                if audio_value.get(key) is not None), None)

        for key, value in (('max_frame', details.get('max_frame')), ('audio', audio_value)):
            try:
                score = float(value)
                details[key] = score if math.isfinite(score) and 0 <= score <= 1 else None
            except (TypeError, ValueError):
                details[key] = None

        try:
            details['fake_frame_count'] = max(0, int(details.get('fake_frame_count', 0)))
        except (TypeError, ValueError):
            details['fake_frame_count'] = None
        if not isinstance(details.get('frame_errors'), list):
            details['frame_errors'] = []
        for key in ('audio_error', 'analysis_error'):
            if details.get(key) is not None and not isinstance(details[key], str):
                details[key] = str(details[key])
        return details

    @property
    def warnings(self):
        if not self.evidence_json:
            return ['Legacy result: not produced by the current detector.']
        warnings = json.loads(self.evidence_json).get('warnings', [])
        if not isinstance(warnings, list):
            return []
        return [
            warning.replace('Reality Defender', 'external detection service')
            for warning in warnings if isinstance(warning, str)
        ]

    @property
    def synthetic_score(self):
        if self.content_type == 'video':
            details = self.video_details
            return details.get('max_frame')
        return json.loads(self.evidence_json).get('fake_score') if self.evidence_json else None

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


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


# ---------- SHARED DETECTOR ADAPTERS ----------

def _run_detection(modality, value):
    try:
        detector = default_detector()
        if modality in {'image', 'text'}:
            return getattr(detector, modality)(value), None
        return detector.file(modality, value), None
    except Exception as exc:
        logging.exception("Detection failed for %s", modality)
        return None, f"Analysis could not be completed: {exc}"


def analyze_image(file_storage):
    image_bytes = file_storage.read()
    suffix = os.path.splitext(file_storage.filename or "")[1] or ".jpg"
    try:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "upload" + suffix)
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            reality_result = detect_image_reality_defender(image_path)
        result = _reality_defender_app_result(reality_result)
        result["raw"] = reality_result.get("raw", reality_result)
        result["deepseek_review"] = _deepseek_image_review(image_bytes)
        return result, None
    except Exception as exc:
        logging.exception("Image analysis failed")
        message = "Image analysis could not be completed."
        result = _reality_defender_app_result({"p_fake": None, "error": message})
        result["raw"] = {"error": str(exc)}
        return result, None


def _reality_defender_app_result(reality_result):
    score = reality_result.get("p_fake")
    message = reality_result.get("error")
    if score is not None:
        try:
            score = float(score)
            if not 0 <= score <= 1:
                raise ValueError("invalid score")
        except (TypeError, ValueError):
            score = None
            message = "The detection service returned an invalid score."

    if score is None:
        status, label, is_threat = "uncertain", "Inconclusive", None
        message = message or "The detection service did not return a score."
        warnings = [message]
    elif score >= REALITY_DEFENDER_FAKE_THRESHOLD:
        status, label, is_threat = "fake", "Likely synthetic / manipulated", True
        warnings = []
    elif score <= REALITY_DEFENDER_REAL_THRESHOLD:
        status, label, is_threat = "real", "Likely authentic", False
        warnings = []
    else:
        status, label, is_threat = "uncertain", "Inconclusive", None
        warnings = []

    if message and any(token in message.casefold() for token in (
        "reality defender", "realitydefender", "reality_defender"
    )):
        message = "The external detection service could not complete the analysis."
        warnings = [message] if score is None else warnings

    return {
        "status": status,
        "label": label,
        "is_threat": is_threat,
        "confidence": score,
        "fake_score": score,
        "calibrated": True,
        "threshold_source": "DeepFake Shield",
        "warnings": warnings,
        "message": message,
    }


def _deepseek_image_review(image_bytes):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "unavailable", "message": "DeepSeek review is not configured."}

    payload = {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Review this image for possible AI generation or digital manipulation. "
                    "Return a concise explanation with: verdict (likely AI, likely real, or uncertain), "
                    "2-4 observable visual indicators, and one limitation. Do not reveal hidden chain-of-thought "
                    "or claim certainty from appearance alone."
                )},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
                }},
            ],
        }],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    request = urllib.request.Request(
        "https://router.huggingface.co/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return {"status": "available", "model": payload["model"], "opinion": content}
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        logging.warning("DeepSeek review unavailable: %s", exc)
        return {"status": "unavailable", "message": "DeepSeek review could not be completed."}


def analyze_text(value):
    return _run_detection('text', value)


def _analyze_upload(modality, upload):
    suffix = os.path.splitext(upload.filename or '')[1]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'upload' + suffix)
        upload.save(path)
        return _run_detection(modality, path)


def analyze_video(upload):
    suffix = os.path.splitext(upload.filename or '')[1]
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'upload' + suffix)
            upload.save(path)
            return detect_video(path), None
    except Exception as exc:
        logging.exception("Video analysis failed")
        return {"analysis_error": str(exc), "frame_scores": [], "frame_errors": [],
                "max_frame": None, "fake_frame_count": 0, "audio": None,
                "audio_error": None}, None


def analyze_audio(upload):
    suffix = os.path.splitext(upload.filename or "")[1]
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "upload" + suffix)
            upload.save(path)
            reality_result = detect_audio_reality_defender(path)
        result = _reality_defender_app_result(reality_result)
        result["raw"] = reality_result.get("raw", reality_result)
        return result, None
    except Exception as exc:
        logging.exception("Audio analysis failed")
        message = "Audio analysis could not be completed."
        result = _reality_defender_app_result({"p_fake": None, "error": message})
        result["raw"] = {"error": str(exc)}
        return result, None


# ---------- PAGE ROUTES ----------

@app.route("/")
def home():
    user = None
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
    return render_template("home.html", user=user)


@app.route('/healthz')
def health_check():
    return jsonify({'status': 'ok'}), 200


@app.route("/login")
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("login.html")


def email_delivery_configured():
    username = app.config.get('SMTP_USERNAME')
    password = app.config.get('SMTP_PASSWORD')
    use_tls = app.config.get('SMTP_USE_TLS')
    use_ssl = app.config.get('SMTP_USE_SSL')
    return bool(
        app.config.get('SMTP_HOST')
        and app.config.get('SMTP_FROM')
        and (bool(username) == bool(password))
        and (use_tls or use_ssl)
        and not (use_tls and use_ssl)
    )


def send_account_email(recipient, subject, body):
    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = app.config['SMTP_FROM']
    message['To'] = recipient
    message.set_content(body)

    host = app.config['SMTP_HOST']
    port = int(app.config['SMTP_PORT'])
    if app.config['SMTP_USE_SSL']:
        smtp = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        smtp = smtplib.SMTP(host, port, timeout=15)
    with smtp:
        if app.config['SMTP_USE_TLS'] and not app.config['SMTP_USE_SSL']:
            smtp.starttls()
        if app.config['SMTP_USERNAME']:
            smtp.login(app.config['SMTP_USERNAME'], app.config['SMTP_PASSWORD'])
        smtp.send_message(message)


def send_password_reset_email(user, token):
    reset_path = url_for('reset_password', token=token)
    reset_url = urljoin(
        app.config['PASSWORD_RESET_BASE_URL'].rstrip('/') + '/', reset_path.lstrip('/')
    )
    send_account_email(
        user.email,
        'Reset your DeepFake Shield password',
        f"Use this link to reset your password. It expires in 60 minutes:\n\n{reset_url}\n\n"
        "If you did not request this, you can ignore this email.",
    )


def send_verification_email(user, token):
    verify_path = url_for('verify_email', token=token)
    verify_url = urljoin(
        app.config['PASSWORD_RESET_BASE_URL'].rstrip('/') + '/', verify_path.lstrip('/')
    )
    send_account_email(
        user.email,
        'Verify your DeepFake Shield email',
        f"Verify your email address within 24 hours by opening this link:\n\n{verify_url}\n\n"
        "If you did not create this account, you can ignore this email.",
    )


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        if not email_delivery_configured():
            flash('Password reset email is not configured. Ask the administrator to configure SMTP.', 'error')
            return render_template('forgot_password.html', mail_configured=False)

        email = (request.form.get('email') or '').strip().lower()
        user = User.query.filter_by(email=email).first() if email else None
        if user:
            token = secrets.token_urlsafe(32)
            user.password_reset_token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
            user.password_reset_expires_at = int(time.time()) + 3600
            db.session.commit()
            try:
                send_password_reset_email(user, token)
            except Exception:
                db.session.rollback()
                user = db.session.get(User, user.id)
                if user:
                    user.password_reset_token_hash = None
                    user.password_reset_expires_at = None
                    db.session.commit()
                logging.exception('Password reset email could not be sent')

        flash('If an account exists for that email, reset instructions will be sent.', 'success')
        return render_template('forgot_password.html', mail_configured=True)

    return render_template(
        'forgot_password.html', mail_configured=email_delivery_configured()
    )


@app.route('/verify-email/<token>')
def verify_email(token):
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    user = User.query.filter_by(email_verification_token_hash=token_hash).first()
    if not user or not user.email_verification_expires_at or user.email_verification_expires_at < int(time.time()):
        flash('This verification link is invalid or expired. Request a new one.', 'error')
        return redirect(url_for('resend_verification'))

    user.email_verified = True
    user.email_verification_token_hash = None
    user.email_verification_expires_at = None
    db.session.commit()
    flash('Email verified. You can now sign in.', 'success')
    return redirect(url_for('login'))


@app.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    mail_configured = email_delivery_configured()
    if request.method == 'POST':
        if not mail_configured:
            flash('Email verification is not configured. Ask the administrator to configure SMTP.', 'error')
            return render_template('resend_verification.html', mail_configured=False)

        email = (request.form.get('email') or '').strip().lower()
        user = User.query.filter_by(email=email).first() if email else None
        if user and not user.email_verified:
            token = secrets.token_urlsafe(32)
            user.email_verification_token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
            user.email_verification_expires_at = int(time.time()) + 86400
            db.session.commit()
            try:
                send_verification_email(user, token)
            except Exception:
                db.session.rollback()
                user = db.session.get(User, user.id)
                if user:
                    user.email_verification_token_hash = None
                    user.email_verification_expires_at = None
                    db.session.commit()
                logging.exception('Verification email could not be sent')

        flash('If that account needs verification, a new link will be sent.', 'success')
        return render_template('resend_verification.html', mail_configured=True)

    return render_template('resend_verification.html', mail_configured=mail_configured)


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    user = User.query.filter_by(password_reset_token_hash=token_hash).first()
    if not user or not user.password_reset_expires_at or user.password_reset_expires_at < int(time.time()):
        flash('This password reset link is invalid or expired. Request a new one.', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        elif password != confirm_password:
            flash('Passwords do not match.', 'error')
        else:
            user.set_password(password)
            user.email_verified = True
            user.email_verification_token_hash = None
            user.email_verification_expires_at = None
            db.session.commit()
            flash('Password reset successfully. Sign in with your new password.', 'success')
            return redirect(url_for('login'))

    return render_template('reset_password.html')


@app.route("/login/google")
def google_login():
    if not app.config.get('GOOGLE_CLIENT_ID') or not app.config.get('GOOGLE_CLIENT_SECRET'):
        flash("Google sign-in is not configured. Add its credentials to .env.", "error")
        return redirect(url_for('login'))
    try:
        return google.authorize_redirect(app.config['GOOGLE_REDIRECT_URI'])
    except Exception as exc:
        logging.warning("Google sign-in could not start: %s", type(exc).__name__)
        flash("Google sign-in could not be started. Please try again.", "error")
        return redirect(url_for('login'))


@app.route("/login/google/callback")
def google_callback():
    try:
        # Authlib verifies the state and validates the OIDC ID token, including its nonce.
        token = google.authorize_access_token()
        identity = token.get('userinfo')
        if not isinstance(identity, dict):
            flash("Google did not return a valid identity.", "error")
            return redirect(url_for('login'))

        email = (identity.get('email') or '').strip().lower()
        google_sub = identity.get('sub')
        email_verified = identity.get('email_verified')
        if not email or not google_sub or not (
            email_verified is True
            or isinstance(email_verified, str) and email_verified.casefold() == 'true'
        ):
            flash("Google must provide a verified email address to sign in.", "error")
            return redirect(url_for('login'))

        user = User.query.filter_by(google_sub=google_sub).first()
        email_user = User.query.filter_by(email=email).first()
        if user and email_user and user.id != email_user.id:
            flash("This Google account conflicts with an existing account. Contact support.", "error")
            return redirect(url_for('login'))
        if user is None:
            user = email_user
            if user and user.google_sub and user.google_sub != google_sub:
                flash("This email is already linked to another Google account.", "error")
                return redirect(url_for('login'))
            if user is None:
                name = (identity.get('name') or email.split('@', 1)[0]).strip()[:120]
                user = User(
                    name=name or email,
                    email=email,
                    password_hash=generate_password_hash(secrets.token_urlsafe(48)),
                    email_verified=True,
                )
                db.session.add(user)

            user.google_sub = google_sub
            user.email_verified = True
            user.email_verification_token_hash = None
            user.email_verification_expires_at = None
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                user = User.query.filter_by(google_sub=google_sub).first()
                if user is None:
                    raise

        session.clear()
        session['user_id'] = user.id
        session['user_name'] = user.name
        return redirect(url_for('dashboard'))
    except Exception as exc:
        db.session.rollback()
        logging.warning("Google sign-in callback failed: %s", type(exc).__name__)
        flash("Google sign-in failed or was cancelled. Please try again.", "error")
        return redirect(url_for('login'))


@app.route("/signup")
def signup():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("signup.html")


@app.route("/dashboard")
@login_required
def dashboard():
    user = db.session.get(User, session['user_id'])

    analyses = (
        Analysis.query
        .filter_by(user_id=user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    total = len(analyses)
    threats = sum(1 for a in analyses if a.is_threat)
    safe = sum(1 for a in analyses if a.is_threat is False)
    uncertain = total - threats - safe
    protection_score = round((safe / total) * 100) if total else 0

    recent = analyses[:5]

    return render_template(
        "dashboard.html",
        user=user,
        total=total,
        safe=safe,
        uncertain=uncertain,
        threats=threats,
        protection_score=protection_score,
        recent=recent,
    )


@app.route("/analyze")
@login_required
def analyze():
    user = db.session.get(User, session['user_id'])
    return render_template("analyze.html", user=user)


@app.route("/history")
@login_required
def history():
    user = db.session.get(User, session['user_id'])

    content_type = request.args.get('type', 'all')
    threat_filter = request.args.get('threat', 'all')

    query = Analysis.query.filter_by(user_id=user.id)

    if content_type != 'all':
        query = query.filter_by(content_type=content_type)

    if threat_filter == 'threat':
        query = query.filter_by(is_threat=True)
    elif threat_filter == 'safe':
        query = query.filter_by(is_threat=False)
    elif threat_filter == 'uncertain':
        query = query.filter(Analysis.is_threat.is_(None))

    analyses = query.order_by(Analysis.created_at.desc()).all()

    return render_template("history.html", user=user, analyses=analyses)


@app.route("/settings")
@login_required
def settings():
    user = db.session.get(User, session['user_id'])
    return render_template("settings.html", user=user)


@app.route("/reports")
@login_required
def reports():
    user = db.session.get(User, session['user_id'])

    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()

    total = len(analyses)
    threats = sum(1 for a in analyses if a.is_threat)
    safe = sum(1 for a in analyses if a.is_threat is False)
    uncertain = total - threats - safe

    image_count = sum(1 for a in analyses if a.content_type == 'image')
    text_count = sum(1 for a in analyses if a.content_type == 'text')

    scores = [a.confidence for a in analyses if a.confidence is not None]
    avg_confidence = round(sum(scores) / len(scores), 2) if scores else 0

    return render_template(
        "reports.html",
        user=user,
        total=total,
        threats=threats,
        safe=safe,
        uncertain=uncertain,
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

    if not email_delivery_configured():
        return jsonify({
            "success": False,
            "message": "Email verification is unavailable. Ask the administrator to configure SMTP.",
        }), 503

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "An account with this email already exists."}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    token = secrets.token_urlsafe(32)
    user.email_verification_token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()
    user.email_verification_expires_at = int(time.time()) + 86400

    db.session.add(user)
    db.session.commit()
    try:
        send_verification_email(user, token)
    except Exception:
        user.email_verification_token_hash = None
        user.email_verification_expires_at = None
        db.session.commit()
        logging.exception('Verification email could not be sent')
        return jsonify({
            "success": False,
            "message": "Could not send the verification email. Request a new link to try again.",
        }), 502

    return jsonify({
        "success": True,
        "message": "Account created. Check your email and verify it before signing in.",
    }), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401

    if not user.email_verified:
        return jsonify({
            "success": False,
            "message": "Verify your email before signing in. Check your inbox or request a new link.",
        }), 403

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

    if content_type == "video":
        analysis_error = result.get("analysis_error")
        verdict = result.get("verdict")
        status = "failed" if analysis_error else (
            verdict if verdict in {"fake", "real", "conflict", "inconclusive"}
            else "inconclusive"
        )
        label = {
            "fake": "Fake",
            "real": "Real",
            "conflict": "Conflict",
            "inconclusive": "Inconclusive",
            "failed": "Analysis failed",
        }[status]
        is_threat = True if status == "fake" else False if status == "real" else None
        confidence = result.get("max_frame")
        fake_score = confidence
        warnings = [analysis_error] if analysis_error else []
        message = analysis_error
        frames_analyzed = len(result.get("frame_scores") or [])
    else:
        status = result["status"]
        label = result["label"]
        is_threat = result["is_threat"]
        confidence = result["confidence"]
        fake_score = result["fake_score"]
        warnings = result["warnings"]
        message = result.get("message")
        frames_analyzed = result.get("frames_analyzed")

    analysis = Analysis(
        user_id=session['user_id'],
        content_type=content_type,
        source_name=source_name,
        image_path=image_path,
        label=label,
        confidence=confidence,
        is_threat=is_threat,
        evidence_json=json.dumps(result, default=str),
    )
    db.session.add(analysis)
    db.session.commit()
    public_result = {key: value for key, value in result.items() if key != "raw"}

    return jsonify({
        "success": True,
        "content_type": content_type,
        "label": label,
        "confidence": confidence,
        "is_threat": is_threat,
        "frames_analyzed": frames_analyzed,
        "status": status,
        "verdict": result.get("verdict") if content_type == "video" else status,
        "fake_score": fake_score,
        "max_frame": result.get("max_frame") if content_type == "video" else None,
        "fake_frame_count": result.get("fake_frame_count") if content_type == "video" else None,
        "audio": result.get("audio") if content_type == "video" else None,
        "frame_errors": result.get("frame_errors", []) if content_type == "video" else [],
        "audio_error": result.get("audio_error") if content_type == "video" else None,
        "message": message,
        "calibrated": result.get("calibrated", False),
        "threshold_source": result.get("threshold_source"),
        "warnings": warnings,
        "evidence": public_result,
    }), 200


@app.route("/api/change-password", methods=["POST"])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}

    user = db.session.get(User, session['user_id'])
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

    user = db.session.get(User, session['user_id'])
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
    user = db.session.get(User, session['user_id'])

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
    user = db.session.get(User, session['user_id'])
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Date', 'Type', 'Source', 'Result', 'Detection score', 'Status',
        'Video verdict', 'Max frame score', 'Fake frame count', 'Audio score',
        'Frame errors', 'Audio error', 'Analysis error',
    ])

    for analysis in analyses:
        details = analysis.video_details
        frame_errors = details.get('frame_errors') or []
        frame_error_text = '; '.join(
            f"{item.get('timestamp', '')}: {item.get('error', '')}"
            for item in frame_errors if isinstance(item, dict)
        )
        video_verdict = ''
        if analysis.content_type == 'video':
            video_verdict = (
                'Analysis failed' if analysis.status == 'failed'
                else details.get('verdict') or 'Inconclusive'
            )
        writer.writerow([
            analysis.created_at.strftime('%Y-%m-%d %H:%M'),
            analysis.content_type.capitalize(),
            analysis.source_name[:50],
            analysis.label,
            analysis.confidence,
            analysis.status,
            video_verdict,
            details.get('max_frame'),
            details.get('fake_frame_count'),
            details.get('audio'),
            frame_error_text,
            details.get('audio_error') or '',
            details.get('analysis_error') or '',
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
    user_columns = {column['name'] for column in inspect(db.engine).get_columns('user')}
    if 'google_sub' not in user_columns:
        db.session.execute(sql_text('ALTER TABLE "user" ADD COLUMN google_sub VARCHAR(255)'))
        db.session.commit()
    if 'email_verified' not in user_columns:
        db.session.execute(sql_text(
            'ALTER TABLE "user" ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT 0'
        ))
        db.session.commit()
    if 'email_verification_token_hash' not in user_columns:
        db.session.execute(sql_text(
            'ALTER TABLE "user" ADD COLUMN email_verification_token_hash VARCHAR(64)'
        ))
        db.session.commit()
    if 'email_verification_expires_at' not in user_columns:
        db.session.execute(sql_text(
            'ALTER TABLE "user" ADD COLUMN email_verification_expires_at INTEGER'
        ))
        db.session.commit()
    if 'password_reset_token_hash' not in user_columns:
        db.session.execute(sql_text('ALTER TABLE "user" ADD COLUMN password_reset_token_hash VARCHAR(64)'))
        db.session.commit()
    if 'password_reset_expires_at' not in user_columns:
        db.session.execute(sql_text('ALTER TABLE "user" ADD COLUMN password_reset_expires_at INTEGER'))
        db.session.commit()
    db.session.execute(sql_text(
        'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_google_sub ON "user" (google_sub)'
    ))
    db.session.commit()
    # Additive migration preserves accounts and existing results.
    if 'evidence_json' not in {column['name'] for column in inspect(db.engine).get_columns('analysis')}:
        db.session.execute(sql_text('ALTER TABLE analysis ADD COLUMN evidence_json TEXT'))
        db.session.commit()


if __name__ == "__main__":
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
