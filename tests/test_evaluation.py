import csv
import pytest
from evaluate_model import read_manifest, reject_leakage, summarize, fit_thresholds


def test_reject_account_metadata(tmp_path):
    path = tmp_path / 'accounts.csv'
    path.write_text('user_id,class_type\n1,human\n')
    with pytest.raises(ValueError, match='Manifest needs'):
        read_manifest(path)


def test_duplicate_content_rejected(tmp_path):
    (tmp_path / 'one.txt').write_text('same text')
    (tmp_path / 'two.txt').write_text('same text')
    path = tmp_path / 'manifest.csv'
    path.write_text('path,modality,label,group,split\none.txt,text,real,a,test\ntwo.txt,text,real,b,test\n')
    with pytest.raises(ValueError, match='Duplicate content'):
        read_manifest(path)


@pytest.mark.parametrize('row', [{'sha256': 'same', 'group': 'new'}, {'sha256': 'new', 'group': 'same'}])
def test_validation_test_leakage_rejected(row):
    with pytest.raises(ValueError, match='overlaps'):
        reject_leakage([row], {'validation_hashes': ['same'], 'validation_groups': ['same']})


def test_metrics_do_not_hide_failures_or_abstention():
    rows = [
        {'modality': 'image', 'label': 'real', 'result': {'status': 'real', 'fake_score': .1}},
        {'modality': 'image', 'label': 'real', 'error': 'failed'},
        {'modality': 'image', 'label': 'fake', 'result': {'status': 'uncertain', 'fake_score': .6}},
        {'modality': 'image', 'label': 'fake', 'result': {'status': 'real', 'fake_score': .1}},
    ]
    report = summarize(rows)['image']
    assert report['total'] == 4
    assert report['errors'] == 1
    assert report['decision_coverage'] == .5
    assert report['selective_accuracy'] == .5
    assert report['correct_decisions_over_all_inputs'] == .25
    assert report['matrix_rows_real_fake_columns_real_fake_uncertain_error'] == [[1,0,0,1],[1,0,1,0]]


def validation_rows():
    return [{'modality': 'image', 'label': label, 'group': f'{label}{i}', 'sha256': f'{label}{i}',
             'split': 'validation', 'result': {'fake_score': score}}
            for label,score in [('real', .02), ('fake', .98)] for i in range(20)]


def test_fit_uses_validation_only_and_requires_independent_groups():
    rows = validation_rows()
    fitted = fit_thresholds(rows)
    assert fitted['thresholds']['image']['real'] == .2
    assert fitted['thresholds']['image']['fake'] == .8
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='validation split'):
        fit_thresholds(rows)
    rows[0]['split'] = 'validation'
    for row in rows:
        row['group'] = 'one_source'
    with pytest.raises(ValueError, match='independent source groups'):
        fit_thresholds(rows)


def test_no_separation_cannot_be_enabled():
    rows = validation_rows()
    for row in rows:
        row['result']['fake_score'] = .99
    with pytest.raises(ValueError, match='No useful'):
        fit_thresholds(rows)
