#test_detection.py
import io
import json
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from detection import Detector, Settings, fake_score, pipeline_id, sample_indices


def test_text_chatgpt_label_is_fake():
    assert fake_score([{"label": "Human", "score": .02}, {"label": "ChatGPT", "score": .98}], "text") == .98


@pytest.mark.parametrize("predictions", [
    [{"label": "LABEL_0", "score": 1}],
    [{"label": "Real", "score": .9}],
    [{"label": "Real", "score": float('nan')}, {"label": "Fake", "score": .1}],
    [{"label": "Real", "score": .9}, {"label": "Fake", "score": .8}],
    [{"label": "Real", "score": .5}, {"label": "Real", "score": .5}],
])
def test_bad_model_responses_never_become_authentic(predictions):
    with pytest.raises(ValueError):
        fake_score(predictions, "image")


def test_audio_label_order_differs_from_image():
    assert fake_score([{"label": "fake", "score": .9}, {"label": "real", "score": .1}], "audio") == .9


@pytest.mark.parametrize("scores,status", [([], "uncertain"), ([.5], "uncertain"), ([.1], "uncertain"), ([.95], "fake"), ([.01, .99, .99, .99], "uncertain")])
def test_abstention(scores, status):
    result = Detector().result("image", scores)
    assert result["status"] == status
    if status == 'uncertain':
        assert result["is_threat"] is None


def test_sampling_includes_last_frame_and_no_duplicates():
    assert sample_indices(1000, 32)[-1] == 999
    assert sample_indices(3, 32) == [0, 1, 2]
    assert sample_indices(0, 32) == []


def test_short_and_unsupported_text_does_not_load_model(monkeypatch):
    monkeypatch.setattr('detection.get_pipeline', lambda _: pytest.fail("Model should not load"))
    assert Detector().text("Very short input")["status"] == "uncertain"
    assert Detector().text("यह एक वाक्य है " * 30)["status"] == "uncertain"


def test_low_resolution_is_inconclusive(monkeypatch):
    monkeypatch.setattr(Detector, "image_score", lambda *args: .99)
    data = io.BytesIO()
    Image.new("RGB", (20,20)).save(data, format="PNG")
    assert Detector().image(data.getvalue())["status"] == "uncertain"


def test_invalid_image_fails():
    with pytest.raises(Exception):
        Detector().image(b"not an image")


def test_thresholds_bound_to_pipeline(tmp_path):
    path = tmp_path / 'thresholds.json'
    path.write_text(json.dumps({"pipeline_id": "wrong", "thresholds": {}}))
    with pytest.raises(ValueError, match="does not match"):
        Detector(calibration_path=path)
    path.write_text(json.dumps({"pipeline_id": pipeline_id(Settings()), "thresholds": {"image": {"real": .05, "fake": .99}}}))
    detector = Detector(calibration_path=path)
    assert detector.result("image", [.9])["status"] == 'uncertain'


def test_audio_checks_later_segments_and_silence(tmp_path, monkeypatch):
    import soundfile as sf
    seen = []
    def fake_pipeline(_):
        def predict(data, **kwargs):
            seen.append(len(data['array']))
            p = .01 if len(seen) == 1 else .99
            return [{'label': 'fake', 'score': p}, {'label': 'real', 'score': 1-p}]
        return predict
    monkeypatch.setattr('detection.get_pipeline', fake_pipeline)
    path = tmp_path / 'audio.wav'
    sf.write(path, np.ones(16000*13, dtype=np.float32)*.1, 16000)
    result = Detector().audio(path)
    assert len(seen) == 3
    assert result['segments'][-1][-1] == 13
    assert result['coverage'] == 1
    assert result['status'] == 'uncertain'
    sf.write(path, np.zeros(16000*2), 16000)
    assert Detector().audio(path)['fake_score'] is None


def test_audio_budget_is_explicit(tmp_path, monkeypatch):
    import soundfile as sf
    monkeypatch.setattr('detection.get_pipeline', lambda _: lambda *a, **k: [{'label': 'fake', 'score': .01}, {'label': 'real', 'score': .99}])
    path = tmp_path / 'audio.wav'
    sf.write(path, np.ones(16000*13)*.1, 16000)
    result = Detector(replace(Settings(), max_audio_segments=2)).audio(path)
    assert result['coverage'] < 1
    assert result['status'] == 'uncertain'


def test_video_tail_and_audio_conflict(tmp_path, monkeypatch):
    import cv2
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100,100))
    for i in range(40):
        writer.write(np.full((100,100,3), i*5, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(Detector, 'image_score', lambda *a: .01)
    monkeypatch.setattr(Detector, 'audio', lambda self, p: self.result('audio', [.99]))
    result = Detector().video(path)
    assert result['frames_analyzed'] == 32
    assert result['timestamps'][-1] == 3.9
    assert result['status'] == 'uncertain'
    assert result['audio']['status'] == 'fake'


def test_text_uses_every_token_and_weights_short_tail(monkeypatch):
    import torch
    from types import SimpleNamespace
    seen = []
    class Tokenizer:
        model_max_length = 12
        def encode(self, text, **kwargs):
            return list(range(81))
        def num_special_tokens_to_add(self, **kwargs):
            return 2
        def decode(self, ids, **kwargs):
            return ' '.join(str(i) for i in ids)
        def __call__(self, text, **kwargs):
            ids = [int(value) for value in text.split()]
            seen.extend(ids)
            return {'input_ids': torch.tensor([ids])}
    class Model:
        config = SimpleNamespace(id2label={0: 'Human', 1: 'ChatGPT'})
        def __call__(self, input_ids):
            p = .5 if input_ids[0,0] == 80 else .01
            return SimpleNamespace(logits=torch.log(torch.tensor([[1-p,p]])))
    classifier = SimpleNamespace(tokenizer=Tokenizer(), model=Model(), device='cpu')
    monkeypatch.setattr('detection.get_pipeline', lambda _: classifier)
    result = Detector().text('word ' * 81)
    assert seen == list(range(81))
    assert result['coverage'] == 1
    assert result['fake_score'] == pytest.approx((80*.01+.5)/81)
    assert result['status'] == 'real'
    seen.clear()
    result = Detector(replace(Settings(), max_text_chunks=2)).text('word ' * 81)
    assert seen[-1] == 80
    assert result['status'] == 'uncertain'
    assert result['coverage'] < 1
