# DeepfakeShield

Flask application for inspecting images, English text, speech audio, and video
for signs of synthetic content. **This is an unvalidated detector, not proof of
authenticity. No accuracy percentage is claimed for this revision.**

## What changed

- Fixed a concrete text-label bug: the model's `ChatGPT` class now maps to
  synthetic. The previous substring check could mark it safe.
- Image, text, audio, and video share the same inference implementation in
  `detection.py`; evaluation no longer uses different preprocessing/models.
- Pinned model revisions and explicit, validated class mappings. Unknown labels,
  malformed scores, missing models, and decoding failures cannot become “real.”
- Local inference for all models, lazy loading, automatic CPU/CUDA selection.
  Content is no longer sent to a hosted inference API. Initial weight downloads
  need internet; subsequent inference can use the local cache.
- Text uses token-sized chunks across the input, weighted by token count.
  Short text and obviously unsupported scripts return inconclusive.
- Speech is decoded to mono 16 kHz and checked in six-second segments across the
  recording, including its end. Silent/very short segments are skipped and
  coverage is recorded. At most 40 distributed segments and ten minutes per file;
  partial sampling is explicitly inconclusive. Split longer recordings.
- Video uses up to 32 evenly distributed frames including the final frame,
  combines full synthetic-content scores, and also checks its audio. Conflicting
  evidence is inconclusive. Audio/visual results remain separate; the unvalidated
  pickle fusion model is no longer loaded.
- Inconclusive results persist as a distinct state in history, reports, filters,
  and CSV exports. They are not counted as authentic. Results remain visible
  instead of disappearing after an automatic page refresh.
- Evaluation reports errors and abstention coverage alongside accuracy, false
  positives, missed fakes, ROC-AUC, and score quality. Duplicate content and
  validation/test source-group overlap are rejected.

## Run locally

Python 3.11–3.13 recommended. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit SECRET_KEY in .env.
.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000 and create an account. First analysis of each modality
downloads its pretrained weights (hundreds of MB per model) into `.cache/`.
FFmpeg comes from `imageio-ffmpeg`; no hardcoded personal Windows path is needed.
The web server starts without loading models. CPU inference can be slow, notably
for 32-frame videos. The development server is intended for local use.

Password signup verification and password reset require SMTP settings in `.env`:
`SMTP_HOST`, `SMTP_PORT`,
`SMTP_USE_TLS` or `SMTP_USE_SSL`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`,
and the public `PASSWORD_RESET_BASE_URL`. Email verification links expire after
24 hours; password-reset links expire after one hour. Both are single-use. Without
SMTP configuration, password signup and the reset/resend forms are unavailable.
Existing password accounts must verify their email once after this migration;
Google sign-in verifies through Google.

Existing SQLite databases receive additive account-verification and
`evidence_json` columns at startup; old results remain labeled as legacy in
history. Back up important databases before upgrading. Old predictions do not
inherit the validity of new ones.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python smoke_models.py
```

Unit/integration tests cover label direction, abstention, full-duration sampling,
decode failures, score validation, API/database behavior, data leakage, and
evaluation denominators. `smoke_models.py` uses actual model weights on generated
fixtures for all four modalities. **These checks are not accuracy benchmarks.**

## Measure performance correctly

The bundled `test_data/text/*.csv` files are account metadata, not passages of
human/AI writing. `test_data/audio/DATASET-balanced.csv` contains spectral features,
not recordings. Neither validates the supplied neural detectors. Supply actual
media/text with independently verified labels and provenance.

Create separate validation and test CSV manifests:

```csv
path,modality,label,group,split
media/photo.jpg,image,real,source-001,test
media/generated.png,image,fake,source-002,test
media/speech.wav,audio,real,speaker-003,test
media/clone.wav,audio,fake,speaker-004,test
media/human.txt,text,real,author-005,test
media/generated.txt,text,fake,prompt-006,test
media/original.mp4,video,real,original-007,test
media/edited.mp4,video,fake,original-008,test
```

These filenames are examples; actual files must exist relative to the manifest.
Keep derivatives of the same source/person/prompt together. Include compressed,
resized, low-quality, edited, and out-of-domain examples. Hold out generators,
speakers, authors, and source videos rather than randomly splitting related
frames. Exclude each model's training data wherever provenance is known.

```powershell
# Baseline of the updated pipeline on independent test data:
.venv\Scripts\python evaluate_model.py test.csv --output evaluation.json
# Optional threshold selection using a DIFFERENT validation-only manifest:
.venv\Scripts\python evaluate_model.py validation.csv --fit-thresholds thresholds.json --output validation-results.json
# Final test; reused files and groups are rejected:
.venv\Scripts\python evaluate_model.py test.csv --thresholds thresholds.json --output evaluation.json
```

Threshold selection requires at least 20 independent groups per class for each
modality and ten supported decisions on each side. This is a minimum guard, not
enough evidence for a strong deployment claim. It only tightens the initial 0.2/
0.8 decision boundaries to satisfy an empirical validation error limit. It rejects
models with no useful separation. Set `DETECTION_THRESHOLDS=thresholds.json` in
`.env` to use the resulting thresholds. They are tied to a pipeline fingerprint.
Threshold selection does **not** calibrate probabilities; scores remain raw.

Always report decision coverage alongside selective accuracy. A system that
abstains on almost everything can have high accuracy on very few decisions.
Failed inputs remain in the report and cause a nonzero evaluator exit status.
Wilson intervals assume independent observations; correlated derivatives violate
that assumption. Do not repeatedly tune against the final test set.

## Models and remaining limits

| Modality | Pinned publisher model | Scope |
|---|---|---|
| Image / video frames | [dima806/deepfake_vs_real_image_detection](https://huggingface.co/dima806/deepfake_vs_real_image_detection) | Visual classifier; no temporal/lip-sync model |
| Text | [Hello-SimpleAI/chatgpt-detector-roberta](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta) | Older English human/ChatGPT detector; not fact checking |
| Speech | [MelodyMachine/Deepfake-audio-detection](https://huggingface.co/MelodyMachine/Deepfake-audio-detection) | Speech model; not validated for music or general sounds |

Model commit hashes and class mappings are recorded in `detection.py`. Merely
adding more classifiers or averaging them does not demonstrate an improvement.
The same pretrained backbones are retained until representative comparisons
justify a replacement. More frames/chunks improve coverage, not necessarily
statistical accuracy. Default uncertainty boundaries and disagreement rules are
engineering heuristics that still require validation.

The image model's publisher explicitly warns of concept drift from older
training data and recommends retraining on recent examples. Its published
historical accuracy does not validate today's uploads. The text model card says
it used all HC3 data for training; do not use HC3 as an independent test set for
this checkpoint. See the publisher links above for provenance.

Modern generators, non-English text (including Latin-script languages), deliberate
paraphrasing, unfamiliar accents, compression, and short/local edits can defeat
these models. The script check is not a language identifier. Silence detection is
not voice-activity detection. A video without usable audio gets visual evidence
only, explicitly noted; a negative frame score cannot prove the whole video real.
Original uploads are stored locally for history. Review their retention and
access policies before exposing this prototype as a public service.
