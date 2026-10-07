import os
import json
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
from unittest.mock import patch
import pytest
import app as web
from detection import Detector


@pytest.fixture()
def client():
    web.app.config.update(TESTING=True, SECRET_KEY='test-key')
    with web.app.app_context():
        web.db.drop_all()
        web.db.create_all()
        user = web.User(name='Test', email='test@example.com', password_hash='unused', email_verified=True)
        web.db.session.add(user)
        web.db.session.commit()
        user_id = user.id
    with web.app.test_client() as client:
        with client.session_transaction() as session:
            session['user_id'] = user_id
        yield client


def test_health_check_is_public(client):
    with client.session_transaction() as session:
        session.clear()
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json == {'status': 'ok'}


def test_uncertain_survives_database_and_all_pages(client):
    response = client.post('/api/analyze', data={'text': 'too short'})
    assert response.status_code == 200
    assert response.json['status'] == 'uncertain'
    assert response.json['is_threat'] is None
    with web.app.app_context():
        record = web.Analysis.query.one()
        assert record.is_threat is None
        assert record.status == 'uncertain'
    for url in ['/dashboard', '/history', '/reports', '/history?threat=uncertain']:
        page = client.get(url)
        assert page.status_code == 200
        assert b'Inconclusive' in page.data or b'inconclusive' in page.data
    assert b'uncertain' in client.get('/api/download-csv').data
    assert b'too short' not in client.get('/history?threat=safe').data


def test_model_failure_is_not_saved_as_authentic(client, monkeypatch):
    class Broken:
        def text(self, value):
            raise RuntimeError('model unavailable')
    monkeypatch.setattr(web, 'default_detector', lambda: Broken())
    response = client.post('/api/analyze', data={'text': 'some content'})
    assert response.status_code == 502
    with web.app.app_context():
        assert web.Analysis.query.count() == 0


@pytest.mark.parametrize('score,status', [(.01,'real'),(.99,'fake'),(.5,'uncertain')])
def test_api_preserves_all_statuses(client, monkeypatch, score, status):
    class Stub:
        def text(self, value):
            return Detector().result('text', [score])
    monkeypatch.setattr(web, 'default_detector', lambda: Stub())
    response = client.post('/api/analyze', data={'text': 'content'})
    assert response.json['status'] == status
    for url in ['/dashboard', '/history', '/reports']:
        assert client.get(url).status_code == 200


def test_email_password_login_and_registration_requires_verification(client, monkeypatch):
    with web.app.app_context():
        user = web.User.query.filter_by(email='test@example.com').one()
        user.set_password('existing-password')
        web.db.session.commit()

    monkeypatch.setitem(web.app.config, 'SMTP_HOST', 'smtp.example.com')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', 'noreply@example.com')
    verification_tokens = []
    monkeypatch.setattr(web, 'send_verification_email', lambda user, token: verification_tokens.append(token))

    login_response = client.post('/api/login', json={
        'email': 'test@example.com', 'password': 'existing-password'
    })
    register_response = client.post('/api/register', json={
        'name': 'New User', 'email': 'new@example.com',
        'password': 'new-password', 'confirmPassword': 'new-password'
    })
    assert login_response.status_code == 200 and login_response.json['success']
    assert register_response.status_code == 201 and register_response.json['success']
    assert len(verification_tokens) == 1

    unverified_login = client.post('/api/login', json={
        'email': 'new@example.com', 'password': 'new-password'
    })
    assert unverified_login.status_code == 403
    assert 'verify your email' in unverified_login.json['message'].lower()

    verified = client.get(f'/verify-email/{verification_tokens[0]}')
    assert verified.status_code == 302
    verified_login = client.post('/api/login', json={
        'email': 'new@example.com', 'password': 'new-password'
    })
    assert verified_login.status_code == 200 and verified_login.json['success']


def test_registration_requires_smtp_configuration(client, monkeypatch):
    monkeypatch.setitem(web.app.config, 'SMTP_HOST', '')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', '')
    response = client.post('/api/register', json={
        'name': 'New User', 'email': 'new@example.com',
        'password': 'new-password', 'confirmPassword': 'new-password'
    })
    assert response.status_code == 503
    with web.app.app_context():
        assert web.User.query.filter_by(email='new@example.com').first() is None


def test_verification_token_expires_and_resend_hides_account_presence(client, monkeypatch):
    import time
    with client.session_transaction() as session:
        session.clear()
    monkeypatch.setitem(web.app.config, 'SMTP_HOST', 'smtp.example.com')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', 'noreply@example.com')
    sent_tokens = []
    monkeypatch.setattr(web, 'send_verification_email', lambda user, token: sent_tokens.append(token))
    response = client.post('/api/register', json={
        'name': 'New User', 'email': 'new@example.com',
        'password': 'new-password', 'confirmPassword': 'new-password'
    })
    assert response.status_code == 201
    token = sent_tokens[-1]
    with web.app.app_context():
        user = web.User.query.filter_by(email='new@example.com').one()
        assert user.email_verified is False
        assert user.email_verification_token_hash != token
        user.email_verification_expires_at = int(time.time()) - 1
        web.db.session.commit()

    expired = client.get(f'/verify-email/{token}', follow_redirects=True)
    assert b'invalid or expired' in expired.data

    known = client.post('/resend-verification', data={'email': 'new@example.com'})
    unknown = client.post('/resend-verification', data={'email': 'missing@example.com'})
    assert b'If that account needs verification' in known.data
    assert b'If that account needs verification' in unknown.data
    assert len(sent_tokens) == 2


def test_password_reset_link_is_delivered_and_one_time(client, monkeypatch):
    with client.session_transaction() as session:
        session.clear()
    with web.app.app_context():
        user = web.User.query.filter_by(email='test@example.com').one()
        user.set_password('old-password')
        web.db.session.commit()

    monkeypatch.setitem(web.app.config, 'SMTP_HOST', 'smtp.example.com')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', 'noreply@example.com')
    sent_tokens = []
    monkeypatch.setattr(web, 'send_password_reset_email', lambda user, token: sent_tokens.append(token))

    response = client.post('/forgot-password', data={'email': 'TEST@example.com'})
    assert response.status_code == 200
    assert b'If an account exists for that email' in response.data
    assert len(sent_tokens) == 1

    token = sent_tokens[0]
    reset_page = client.get(f'/reset-password/{token}')
    assert reset_page.status_code == 200
    reset_response = client.post(f'/reset-password/{token}', data={
        'password': 'new-password', 'confirm_password': 'new-password'
    })
    assert reset_response.status_code == 302
    with web.app.app_context():
        user = web.User.query.filter_by(email='test@example.com').one()
        assert user.check_password('new-password')
        assert user.password_reset_token_hash is None
        assert user.password_reset_expires_at is None

    replay_response = client.get(f'/reset-password/{token}', follow_redirects=True)
    assert b'invalid or expired' in replay_response.data


def test_password_reset_does_not_disclose_unknown_email(client, monkeypatch):
    with client.session_transaction() as session:
        session.clear()
    monkeypatch.setitem(web.app.config, 'SMTP_HOST', 'smtp.example.com')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', 'noreply@example.com')
    sent = []
    monkeypatch.setattr(web, 'send_password_reset_email', lambda user, token: sent.append(user.email))

    known = client.post('/forgot-password', data={'email': 'test@example.com'})
    unknown = client.post('/forgot-password', data={'email': 'missing@example.com'})
    assert b'If an account exists for that email' in known.data
    assert b'If an account exists for that email' in unknown.data
    assert sent == ['test@example.com']


def test_password_reset_rejects_expired_token(client, monkeypatch):
    import time
    with client.session_transaction() as session:
        session.clear()
    monkeypatch.setitem(web.app.config, 'SMTP_HOST', 'smtp.example.com')
    monkeypatch.setitem(web.app.config, 'SMTP_FROM', 'noreply@example.com')
    sent_tokens = []
    monkeypatch.setattr(web, 'send_password_reset_email', lambda user, token: sent_tokens.append(token))
    client.post('/forgot-password', data={'email': 'test@example.com'})
    token = sent_tokens[0]
    with web.app.app_context():
        user = web.User.query.filter_by(email='test@example.com').one()
        user.password_reset_expires_at = int(time.time()) - 1
        web.db.session.commit()

    response = client.get(f'/reset-password/{token}', follow_redirects=True)
    assert b'invalid or expired' in response.data


def test_google_login_button_and_local_callback(client, monkeypatch):
    with client.session_transaction() as session:
        session.clear()
    response = client.get('/login')
    assert b'Continue with Google' in response.data
    assert b'loginForm' in response.data

    monkeypatch.setitem(web.app.config, 'GOOGLE_CLIENT_ID', 'test-client-id')
    monkeypatch.setitem(web.app.config, 'GOOGLE_CLIENT_SECRET', 'test-client-secret')
    with patch.object(web.google, 'authorize_redirect', return_value=web.redirect('/consent')) as authorize:
        response = client.get('/login/google', base_url='http://localhost:5000')
    assert response.status_code == 302
    authorize.assert_called_once_with('http://localhost:5000/login/google/callback')
    assert '/login/google/callback' in {rule.rule for rule in web.app.url_map.iter_rules()}


def test_google_links_existing_user_by_verified_email(client, monkeypatch):
    identity = {
        'sub': 'google-subject-123',
        'email': 'TEST@example.com',
        'email_verified': True,
        'name': 'Test Account',
    }
    monkeypatch.setattr(web.google, 'authorize_access_token', lambda: {'userinfo': identity})

    response = client.get('/login/google/callback')
    assert response.status_code == 302
    with web.app.app_context():
        assert web.User.query.count() == 1
        user = web.User.query.filter_by(email='test@example.com').one()
        assert user.google_sub == 'google-subject-123'
    with client.session_transaction() as session:
        assert session['user_id'] == user.id


def test_google_creates_one_user_and_rejects_unverified_email(client, monkeypatch):
    verified_identity = {
        'sub': 'new-google-subject',
        'email': 'new-google@example.com',
        'email_verified': 'true',
        'name': 'New Google User',
    }
    monkeypatch.setattr(web.google, 'authorize_access_token', lambda: {'userinfo': verified_identity})
    client.get('/login/google/callback')
    client.get('/login/google/callback')
    with web.app.app_context():
        assert web.User.query.count() == 2
        created = web.User.query.filter_by(google_sub='new-google-subject').one()
        assert created.email_verified is True
        created = web.User.query.filter_by(google_sub='new-google-subject').one()
        assert created.email == 'new-google@example.com'

    with client.session_transaction() as session:
        session.clear()
    unverified_identity = dict(verified_identity, sub='unverified-subject',
                               email='unverified@example.com', email_verified=False)
    monkeypatch.setattr(web.google, 'authorize_access_token', lambda: {'userinfo': unverified_identity})
    response = client.get('/login/google/callback', follow_redirects=True)
    assert b'verified email address' in response.data
    with web.app.app_context():
        assert web.User.query.count() == 2


def test_google_callback_rejects_invalid_oauth_state(client):
    with client.session_transaction() as session:
        session.clear()
    response = client.get(
        '/login/google/callback?code=not-a-real-code&state=attacker-state',
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b'Google sign-in failed or was cancelled' in response.data
    with web.app.app_context():
        assert web.User.query.count() == 1


def test_video_pages_handle_legacy_audio_result_objects(client):
    with web.app.app_context():
        user = web.User.query.filter_by(email='test@example.com').one()
        record = web.Analysis(
            user_id=user.id,
            content_type='video',
            source_name='legacy-video.mp4',
            label='Inconclusive',
            confidence=None,
            is_threat=None,
            evidence_json=json.dumps({
                'verdict': 'inconclusive',
                'max_frame': None,
                'fake_frame_count': 0,
                'audio': {'p_fake': None, 'error': 'no score returned'},
                'frame_scores': [None],
                'frame_errors': [],
            }),
        )
        web.db.session.add(record)
        web.db.session.commit()

    for route in ('/dashboard', '/history', '/reports'):
        response = client.get(route)
        assert response.status_code == 200
        assert b'undefined' not in response.data
        assert b'null' not in response.data


def test_existing_database_migrates_without_losing_history(tmp_path):
    import sqlite3
    import subprocess
    import sys
    from pathlib import Path
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as database:
        database.execute('CREATE TABLE user (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at DATETIME)')
        database.execute("INSERT INTO user (id, name, email, password_hash) VALUES (1, 'Existing User', 'existing@example.com', 'existing-hash')")
        database.execute('CREATE TABLE analysis (id INTEGER PRIMARY KEY, user_id INTEGER, content_type TEXT, source_name TEXT, image_path TEXT, label TEXT, confidence FLOAT, is_threat BOOLEAN, created_at DATETIME)')
        database.execute("INSERT INTO analysis (id, content_type, source_name, label, confidence, is_threat) VALUES (1, 'text', 'original record', 'Human', 95, 0)")
    env = dict(os.environ, DATABASE_URL='sqlite:///' + path.as_posix())
    process = subprocess.run([sys.executable, '-c', 'import app'], cwd=Path(web.__file__).parent,
                             env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    with sqlite3.connect(path) as database:
        user_columns = {r[1] for r in database.execute('PRAGMA table_info(user)')}
        assert 'google_sub' in user_columns
        assert 'email_verified' in user_columns
        assert 'email_verification_token_hash' in user_columns
        assert 'email_verification_expires_at' in user_columns
        assert database.execute('SELECT name, email, password_hash FROM user WHERE id=1').fetchone() == ('Existing User', 'existing@example.com', 'existing-hash')
        assert database.execute('SELECT email_verified FROM user WHERE id=1').fetchone()[0] == 0
        assert 'evidence_json' in {r[1] for r in database.execute('PRAGMA table_info(analysis)')}
        assert database.execute('SELECT source_name FROM analysis WHERE id=1').fetchone()[0] == 'original record'
