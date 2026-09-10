"""
train_fusion_model.py

Trains a small "fusion" classifier that combines a video's frame-based
deepfake score with its own audio track's voice-deepfake score into a
single combined verdict.

WHY
---
analyze_video() (frames) and analyze_audio() (voice) each only look at one
modality. A manipulated video might have a faked face but an untouched
voice, or vice versa. Combining both scores into one small trained model
captures signal neither modality gives alone — this is the "fusion" /
"ensemble voting" layer.

TRAINING DATA
-------------
Reuses the same test_data/video/real and test_data/video/fake folders you
built for evaluate_model.py's confusion matrix — every video in there
already has both a picture and a soundtrack, so no extra data collection
is needed.

HONEST CAVEAT
-------------
With a handful of hand-collected videos, this is a small, illustrative
fusion model — good enough to demonstrate the concept works end-to-end,
not a production-grade classifier. Re-run this script any time you add
more labeled videos to test_data/video/.

SETUP
-----
    pip install scikit-learn joblib moviepy opencv-python-headless

RUN
---
    python train_fusion_model.py

Saves fusion_model.pkl into the project folder. app.py picks it up
automatically on the next restart and uses it for video analysis
(falls back to frame-only voting if the file isn't there).
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Reuse the exact same classification helpers evaluate_model.py and app.py
# use, so the features here match what the live app computes.
from evaluate_model import classify_image_bytes, classify_audio_bytes, VIDEO_FRAMES_TO_SAMPLE

TEST_DATA_DIR = Path(__file__).parent / "test_data" / "video"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi"}


def video_frame_score(path):
    """Fraction of sampled frames the image model classified as 'fake' (0.0-1.0)."""
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return None

    sample_count = min(VIDEO_FRAMES_TO_SAMPLE, total_frames)
    frame_indices = [int(total_frames * i / sample_count) for i in range(sample_count)]

    votes = []
    for idx in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = capture.read()
        if not success:
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        label = classify_image_bytes(buffer.tobytes(), "image/jpeg")
        if label:
            votes.append(label)

    capture.release()
    if not votes:
        return None
    return votes.count("fake") / len(votes)


def audio_track_score(path):
    """1.0 if the video's audio track is flagged fake, 0.0 if real, None on failure."""
    try:
        clip = VideoFileClip(str(path))
        if clip.audio is None:
            clip.close()
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        clip.audio.write_audiofile(tmp_path, logger=None)
        clip.close()

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        os.remove(tmp_path)

        label = classify_audio_bytes(audio_bytes, "audio/wav")
        return None if label is None else (1.0 if label == "fake" else 0.0)

    except Exception as exc:
        print(f"    Audio extraction failed: {exc}")
        return None


def collect_videos(folder):
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]


def main():
    real_videos = collect_videos(TEST_DATA_DIR / "real")
    fake_videos = collect_videos(TEST_DATA_DIR / "fake")

    if len(real_videos) + len(fake_videos) < 6:
        print("Need at least a handful of videos in test_data/video/real and "
              "test_data/video/fake (10+ each is better) before training anything meaningful.")
        return

    X, y = [], []

    for path, true_label in [(p, "real") for p in real_videos] + [(p, "fake") for p in fake_videos]:
        print(f"Processing [{true_label}] {path.name}...")

        v_score = video_frame_score(path)
        a_score = audio_track_score(path)

        if v_score is None or a_score is None:
            print("  -> skipped (couldn't extract a score)")
            continue

        X.append([v_score, a_score])
        y.append(1 if true_label == "fake" else 0)
        print(f"  -> video_fake_fraction={v_score:.2f}, audio_fake={a_score}")

    if len(set(y)) < 2:
        print("Need both real and fake examples with successful scores to train. Stopping.")
        return

    X = np.array(X)
    y = np.array(y)

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        # too little data for a meaningful held-out split — train on everything
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\nTrained on {len(X_train)} samples, evaluated on {len(X_test)} samples.")
    print(f"Train accuracy: {train_acc * 100:.1f}%")
    print(f"Test accuracy: {test_acc * 100:.1f}%")
    print(
        f"Learned weights -> video: {model.coef_[0][0]:.3f}, "
        f"audio: {model.coef_[0][1]:.3f}, bias: {model.intercept_[0]:.3f}"
    )

    joblib.dump(model, "fusion_model.pkl")
    print("\nSaved fusion_model.pkl — app.py will use it automatically for video analysis.")


if __name__ == "__main__":
    main()