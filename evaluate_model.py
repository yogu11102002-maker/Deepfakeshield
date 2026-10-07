"""Evaluate the production detector using a labeled CSV manifest.

python evaluate_model.py test.csv --output evaluation.json
python evaluate_model.py validation.csv --fit-thresholds thresholds.json
python evaluate_model.py test.csv --thresholds thresholds.json

Columns: path,modality,label,group,split. Labels: real/fake. Splits:
validation/test. Group identifies the original source, speaker, or author;
related derivatives must share a group. Paths are relative to the manifest.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from detection import Detector, Settings, pipeline_id


def read_manifest(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"path", "modality", "label", "group", "split"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Manifest needs path,modality,label,group,split columns. Account metadata and audio-feature CSVs are not supported.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    hashes, group_splits = set(), {}
    for row in rows:
        if row["modality"] not in {"image", "text", "audio", "video"} or row["label"] not in {"real", "fake"}:
            raise ValueError("Invalid modality or real/fake label")
        if not row["group"].strip() or row["split"] not in {"validation", "test"}:
            raise ValueError("Each row needs a source group and validation/test split")
        group = row["group"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError(f"Source group crosses validation/test splits: {group}")
        group_splits[group] = row["split"]
        resolved = (path.parent / row["path"]).resolve()
        with resolved.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate content detected: {row['path']}")
        hashes.add(digest)
        row.update(resolved_path=str(resolved), sha256=digest)
    return rows


def reject_leakage(rows, calibration):
    known_hashes = set(calibration["validation_hashes"])
    known_groups = set(calibration["validation_groups"])
    if any(r["sha256"] in known_hashes or r["group"] in known_groups for r in rows):
        raise ValueError("Test data overlaps threshold-fitting data (content or source group)")


def wilson(successes, total):
    if not total:
        return None
    p, z = successes/total, 1.96
    center = (p + z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [float(center-half), float(center+half)]


def summarize(rows):
    """Report coverage alongside selective accuracy; failures remain visible."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    reports = {}
    for modality in sorted({r["modality"] for r in rows}):
        selected = [r for r in rows if r["modality"] == modality]
        scored = [r for r in selected if r.get("result", {}).get("fake_score") is not None]
        decided = [r for r in selected if r.get("result", {}).get("status") in {"real", "fake"}]
        correct = sum(r["result"]["status"] == r["label"] for r in decided)
        matrix = [[0]*4 for _ in range(2)]
        for row in selected:
            status = row.get("result", {}).get("status", "error")
            matrix[row["label"] == "fake"][["real", "fake", "uncertain", "error"].index(status)] += 1
        real_total, fake_total = map(sum, matrix)
        predicted_fake = matrix[0][1] + matrix[1][1]
        precision = matrix[1][1]/predicted_fake if predicted_fake else None
        recall = matrix[1][1]/fake_total if fake_total else None
        report = {
            "total": len(selected), "scored": len(scored), "decided": len(decided),
            "errors": sum("error" in r for r in selected),
            "decision_coverage": len(decided)/len(selected),
            "selective_accuracy": correct/len(decided) if decided else None,
            "selective_accuracy_wilson95": wilson(correct, len(decided)),
            "correct_decisions_over_all_inputs": correct/len(selected),
            "false_positive_rate_over_all_real": matrix[0][1]/real_total if real_total else None,
            "false_negative_rate_over_all_fake": matrix[1][0]/fake_total if fake_total else None,
            "fake_detection_rate_over_all_fake": matrix[1][1]/fake_total if fake_total else None,
            "fake_precision": precision,
            "fake_recall_over_all_fake": recall,
            "fake_f1": 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None,
            "matrix_rows_real_fake_columns_real_fake_uncertain_error": matrix,
        }
        if scored:
            y = [int(r["label"] == "fake") for r in scored]
            p = [r["result"]["fake_score"] for r in scored]
            report["raw_score_accuracy_at_0_5"] = float(np.mean([(s >= .5) == label for s,label in zip(p,y)]))
            report["brier_score_uncalibrated"] = float(brier_score_loss(y, p))
            if len(set(y)) == 2:
                report["roc_auc"] = float(roc_auc_score(y, p))
                report["average_precision"] = float(average_precision_score(y, p))
        reports[modality] = report
    return reports


def fit_thresholds(rows, max_error=0.05):
    """Select thresholds on validation only. Error limits are empirical,
    not statistical guarantees. Never evaluate on the training set."""
    if any(r["split"] != "validation" for r in rows):
        raise ValueError("Threshold fitting accepts only the validation split")
    thresholds = {}
    for modality in sorted({r["modality"] for r in rows}):
        examples = [r for r in rows if r["modality"] == modality]
        if any("error" in r or r.get("result", {}).get("fake_score") is None for r in examples):
            raise ValueError(f"Fix failed or unscorable {modality} validation inputs before fitting")
        groups = [{r["group"] for r in examples if r["label"] == label} for label in ("real", "fake")]
        if min(map(len, groups)) < 20:
            raise ValueError(f"Need at least 20 independent source groups per class for {modality} validation")
        real = np.array([r["result"]["fake_score"] for r in examples if r["label"] == "real"])
        fake = np.array([r["result"]["fake_score"] for r in examples if r["label"] == "fake"])
        # Tighten conservative defaults only. Require useful support on both sides.
        lows = [x for x in np.linspace(0, .2, 201) if np.mean(fake <= x) <= max_error and np.sum(real <= x) >= 10]
        highs = [x for x in np.linspace(.8, 1, 201) if np.mean(real >= x) <= max_error and np.sum(fake >= x) >= 10]
        if not lows or not highs:
            raise ValueError(f"No useful conservative thresholds for {modality}; improve model/data rather than force classification")
        thresholds[modality] = {"real": float(max(lows)), "fake": float(min(highs)),
                               "validation_real": len(real), "validation_fake": len(fake),
                               "target_empirical_error": max_error}
    return {"pipeline_id": pipeline_id(Settings()), "thresholds": thresholds,
            "validation_hashes": sorted({r["sha256"] for r in rows}),
            "validation_groups": sorted({r["group"] for r in rows}),
            "note": "Threshold selection only; scores remain uncalibrated. Evaluate on independent test data."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--fit-thresholds", type=Path)
    parser.add_argument("--max-error", type=float, default=.05)
    args = parser.parse_args()
    if args.thresholds and args.fit_thresholds:
        parser.error("Fitting and test evaluation are separate operations")
    if not 0 <= args.max_error <= .2:
        parser.error("--max-error must be between 0 and 0.2")
    rows = read_manifest(args.manifest)
    expected = "validation" if args.fit_thresholds else "test"
    if any(row["split"] != expected for row in rows):
        parser.error(f"This run requires a {expected}-only manifest")
    if args.thresholds:
        reject_leakage(rows, json.loads(args.thresholds.read_text(encoding="utf-8")))
    detector = Detector(calibration_path=args.thresholds)
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['modality']}: {row['path']}", flush=True)
        try:
            row["result"] = detector.file(row["modality"], row["resolved_path"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    report = {"pipeline_id": pipeline_id(detector.settings), "split": expected,
              "metrics": summarize(rows), "predictions": rows,
              "limitations": "Intervals assume independent observations. Group related derivatives; exclude model training data. No guarantee for new generators/domains."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    if args.fit_thresholds:
        args.fit_thresholds.write_text(json.dumps(fit_thresholds(rows, args.max_error), indent=2), encoding="utf-8")
    if any("error" in row for row in rows):
        raise SystemExit("Evaluation incomplete: failed inputs are recorded in the report")


if __name__ == "__main__":
    main()
