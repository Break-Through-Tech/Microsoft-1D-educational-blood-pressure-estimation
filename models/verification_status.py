"""Persistent verification attempts and a separate, checked latest outcome."""
import inspect
import json
from pathlib import Path
import time
import traceback
from uuid import uuid4

from models.comparison_training import digest, write_json


class VerificationError(RuntimeError):
    pass


def run_verification(output, verifier):
    """Archive every outcome; exceptions and incomplete results remain fatal."""
    output = Path(output)
    started = time.time()
    attempt_id = f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))}_{uuid4().hex}'
    record = output / 'verification_attempts' / attempt_id / 'result.json'
    freeze = output / 'selections_frozen.json'
    try:
        source = inspect.getsourcefile(verifier)
    except TypeError:
        source = None
    data = dict(attempt_id=attempt_id, started_unix=started, status='running',
        selection_sha256=digest(freeze) if freeze.exists() else None,
        verifier_sha256=digest(source) if source and Path(source).is_file() else None)

    def save():
        write_json(record, data)
        write_json(output / 'verification_status.json', dict(attempt_id=attempt_id,
            status=data['status'], record=str(record.relative_to(output)), record_sha256=digest(record)))

    save()
    try:
        result = verifier(output)
        data['result'] = result
        if not isinstance(result, dict) or result.get('verified') is not True or result.get('incomplete'):
            raise VerificationError('Verification unsuccessful or selected models incomplete')
    except Exception as error:
        data.update(status='failed', ended_unix=time.time(), error=repr(error), traceback=traceback.format_exc())
        save()
        raise
    else:
        data.update(status='passed', ended_unix=time.time())
        save()
        return result


def require_successful_verification(output):
    """Read new status or legacy evidence without migrating existing artifacts."""
    output = Path(output)
    status_path = output / 'verification_status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        record_path = output / status['record']
        if not record_path.resolve().is_relative_to((output / 'verification_attempts').resolve()):
            raise VerificationError('Verification record is outside attempt history')
        if digest(record_path) != status['record_sha256']:
            raise VerificationError('Verification attempt record changed')
        record = json.loads(record_path.read_text())
        if status['status'] != 'passed' or record['status'] != 'passed' or record['attempt_id'] != status['attempt_id']:
            raise VerificationError('Latest verification attempt did not pass')
        result = record.get('result', {})
        freeze = output / 'selections_frozen.json'
        if record.get('selection_sha256') != (digest(freeze) if freeze.exists() else None):
            raise VerificationError('Verified selection manifest changed')
    else:
        if (output / 'independent_verification_failure.json').exists():
            raise VerificationError('Unresolved legacy verification failure')
        result = json.loads((output / 'independent_verification.json').read_text())
    if result.get('verified') is not True or result.get('incomplete'):
        raise VerificationError('Verification unsuccessful or selected models incomplete')
    return result
