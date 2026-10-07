"""Integration smoke check; generated fixtures do NOT measure accuracy.

Downloads pinned model weights on first run. Run: python smoke_models.py
"""
import io
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from detection import Detector, Settings


def main():
    import cv2
    import soundfile as sf
    detector = Detector(Settings(frame_count=3))
    checks = {}
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        image = Image.new("RGB", (224, 224), (100, 140, 180))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        print("Checking actual image model...", flush=True)
        checks["image"] = detector.image(encoded.getvalue())
        print("Checking actual text model...", flush=True)
        paragraph = "This is a software integration test, not an example of a verified human author. The system should read the complete passage, map both class labels correctly, and return a finite model score. A model score is not evidence of measured accuracy. We use this passage only to confirm that the tokenizer, neural network, and application can communicate. Scientific evaluation requires independent examples with reliable labels and a clear account of where those examples came from. No conclusion about detection quality should be drawn from this generated test passage."
        checks["text"] = detector.text(paragraph)
        print("Checking actual audio model on a non-speech fixture...", flush=True)
        wave = .1*np.sin(2*np.pi*220*np.arange(32000)/16000)
        audio = folder / "tone.wav"
        sf.write(audio, wave, 16000)
        checks["audio"] = detector.audio(audio)
        print("Checking video decoding and model integration...", flush=True)
        video = folder / "fixture.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (224, 224))
        for _ in range(5):
            writer.write(np.full((224,224,3), 120, dtype=np.uint8))
        writer.release()
        checks["video"] = detector.video(video)
    for modality, result in checks.items():
        assert result["fake_score"] is not None and 0 <= result["fake_score"] <= 1, modality
    output = Path("smoke-results.json")
    output.write_text(json.dumps({"purpose": "Integration only; not an accuracy benchmark", "results": checks}, indent=2), encoding="utf-8")
    print("All four model integration checks passed.", flush=True)


if __name__ == "__main__":
    main()
