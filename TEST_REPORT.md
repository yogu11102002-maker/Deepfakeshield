# Verification report

Date: 2026-09-16

Source: `yogu11102002-maker/Deepfakeshield`, main commit
`706c876a602bddf77653eff0c7282e8cef8ddb0b`.

## Completed checks

- Python 3.13.1 on Windows, CPU inference.
- **34 automated tests passed**, including exact class mapping, invalid model
  responses, inconclusive persistence, API rendering, failure handling, late
  audio/video coverage, token chunking, weighted text aggregation, source-group
  leakage, and additive migration of an existing database.
- **Actual model inference completed for image, text, audio, and video** using
  the pinned public weights. Image preprocessing, tokenizer batching, audio
  decoding, frame extraction, and model result conversion all executed.
- `node --check static/js/script.js`: passed.
- `python -m pip check`: no broken requirements.
- `git diff --check`: no whitespace errors.
- Exact installed versions are listed in `requirements-lock.txt`. Model weights
  and the local environment are excluded from the delivery archive.

## Accuracy was NOT established

The source repository does not contain a usable, independently labeled test set
for these neural detectors. Its text CSVs contain account metadata; its audio CSV
contains extracted features, not recordings. Neither was used to invent a result.

The real-model smoke checks use generated fixtures solely to exercise code paths.
They are not representative benchmarks. In particular, the AI-written test
passage was labeled likely authentic by the legacy text checkpoint (synthetic
score about 0.0011). **This is an observed false negative**, and demonstrates that
correct software operation does not guarantee accurate detection. The solid-color
image/video and sine-wave audio fixtures are also outside representative detector
data; their classifications are not evidence of quality.

No before/after benchmark, fine-tuning, probability calibration, or independent
accuracy measurement was completed. The changes fix known code defects and
increase content coverage. Whether those changes improve overall accuracy,
precision, or recall on real uploads remains to be measured. Abstention can
reduce confident mistakes but also reduces decision coverage; report both.

## Remaining work needed for an accuracy claim

1. Collect independently labeled recent images, videos, English writing, and
   speech recordings representative of intended uploads. Document provenance.
2. Hold out sources, speakers, authors, prompts, and generators. Do not evaluate
   the text checkpoint on HC3, which its publisher says was used in training.
3. Compare models and preprocessing on validation data. Check false alarms and
   missed fakes separately, including compression, accents, paraphrasing, and
   short manipulated segments. Retrain or replace models that fail these checks.
4. Select thresholds on validation data only; use the supplied evaluator's
   duplicate-content and group-overlap protections.
5. Evaluate once on an untouched test set and report decision coverage, failures,
   per-modality metrics, and uncertainty intervals. Keep monitoring new generators.

The current default thresholds are engineering heuristics, not calibrated
probabilities. The image publisher explicitly warns of concept drift in its
[model card](https://huggingface.co/dima806/deepfake_vs_real_image_detection/blob/main/README.md).
The [text model card](https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta/blob/main/README.md)
documents its HC3 training scope. The [audio model card](https://huggingface.co/MelodyMachine/Deepfake-audio-detection/blob/main/README.md)
does not provide enough provenance to infer broad generalization from its reported
metrics. These publisher figures are not app accuracy.

## Delivery status

Updated source is packaged for local use. GitHub has not been modified or pushed.
Setup and evaluation commands are in `README.md`.
