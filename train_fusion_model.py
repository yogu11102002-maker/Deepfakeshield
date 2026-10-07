"""The legacy unvalidated fusion trainer has been retired.

It imported a nonexistent function, used different features from the app,
and evaluated on training data when samples were scarce. Old .pkl files are
not loaded. Use validation-only threshold selection and independent testing:

python evaluate_model.py validation.csv --fit-thresholds thresholds.json
python evaluate_model.py test.csv --thresholds thresholds.json

Audio and visual evidence stay separate until paired, source-disjoint video
data demonstrates that learned fusion improves held-out performance.
"""
if __name__ == "__main__":
    raise SystemExit(__doc__)
