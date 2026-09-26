import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / '.artifacts/gguf/release'
VERIFY = HERE.parent / '.artifacts/gguf/remote-verify'
REPO = 'Ninnix96/Qwengram-2b'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(token):
    manifest = json.loads((RELEASE / 'SHA256.json').read_text())
    definition = json.loads((HERE.parent / 'qwengram-2b.json').read_text())
    assert definition == manifest['canonical']
    assert definition['recipe'] == 'REAL-10M + linear750'
    runtime = json.loads((HERE / 'results.json').read_text())
    assert runtime['reader_gain_95ci']['q8'][0] > 0
    for name, spec in manifest['files'].items():
        path = RELEASE / name
        assert path.stat().st_size == spec['size'] and sha(path) == spec['sha256'], name
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ninnix96'
    api.create_repo(REPO, private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=REPO, folder_path=RELEASE,
                               commit_message='Release Qwengram-2B balanced 10M GGUFs and matched validation')
    info = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    verified = {}
    for name, spec in {**manifest['files'], 'SHA256.json': {
            'sha256': sha(RELEASE / 'SHA256.json'), 'size': (RELEASE / 'SHA256.json').stat().st_size}}.items():
        entry = remote[name]
        assert entry.size == spec['size'], name
        if entry.lfs:
            assert entry.lfs.sha256 == spec['sha256'], name
            method = 'HF content-addressed LFS SHA256 and byte size'
        else:
            local = hf_hub_download(REPO, name, revision=commit.oid, token=token, local_dir=VERIFY)
            assert sha(local) == spec['sha256'], name
            method = 'Downloaded content SHA256 and byte size'
        verified[name] = {**spec, 'verification': method}
    if info.private:
        api.update_repo_visibility(REPO, private=False)
    public = HfApi(token=False).model_info(REPO, revision=commit.oid)
    assert not public.private
    result = {'repo': REPO, 'url': 'https://huggingface.co/' + REPO,
              'commit': commit.oid, 'public': True, 'verified_files': verified}
    (HERE / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'verified_files'}, indent=2))
    print('Verified', len(verified), 'published files')


if __name__ == '__main__':
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise SystemExit('Set HF_TOKEN before publishing the release')
    try:
        main(token)
    except Exception as error:
        raise SystemExit(str(error).replace(token, '[REDACTED]')) from None
