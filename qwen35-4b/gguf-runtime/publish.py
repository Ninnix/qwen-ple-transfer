import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / '.artifacts/gguf/release'
VERIFY = HERE.parent / '.artifacts/gguf/remote-verify'
REPO = 'Ninnix96/Qwengram-4B'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main(token, upload, verify_commit):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ninnix96'
    try:
        before = api.model_info(REPO)
    except RepositoryNotFoundError:
        before = None
    print(json.dumps({'repo': REPO, 'exists': before is not None,
                      'commit': before.sha if before else None}), flush=True)
    if not upload and not verify_commit:
        return
    manifest = json.loads((RELEASE / 'SHA256.json').read_text())
    definition = json.loads((HERE.parent / 'qwengram-4b.json').read_text())
    assert definition == manifest['canonical']
    assert definition['recipe'] == 'REAL-15M + linear750'
    assert definition['reader_tokens'] == 15000064
    assert manifest['files']['reader.safetensors']['sha256'] == definition['reader_sha256']
    assert manifest['files']['arbiter.pt']['sha256'] == definition['arbiter_sha256']
    runtime = json.loads((HERE / 'results.json').read_text())
    assert runtime['canonical'] == definition
    for key, precision in (('bf16', 'BF16'), ('q8', 'Q8_0'), ('q6', 'Q6_K'), ('q4', 'Q4_K_M')):
        assert runtime['reader_gain_95ci'][key][0] > 0
        name = 'QwenGram-4B-%s.gguf' % precision
        assert runtime['verification'][precision]['qwengram_sha256'] == manifest['files'][name]['sha256']
    for name, spec in manifest['files'].items():
        path = RELEASE / name
        assert path.stat().st_size == spec['size'] and sha(path) == spec['sha256'], name
    if verify_commit:
        oid = verify_commit
    else:
        if before is None:
            api.create_repo(REPO, repo_type='model', private=False)
            before = api.model_info(REPO)
        assert not before.private
        assert {s.rfilename for s in before.siblings} <= {'.gitattributes'}, 'Review existing repository before replacement'
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=RELEASE / name)
                      for name in [*manifest['files'], 'SHA256.json']]
        commit = api.create_commit(repo_id=REPO, parent_commit=before.sha, operations=operations,
                                  commit_message='Release Qwengram 4B REAL-15M plus linear750',
                                  commit_description='Verified BF16, Q8_0, Q6_K and Q4_K_M with frozen evaluation, matched runtime retention, sidecar checks and artifact hashes.')
        oid = commit.oid
        print('Published', commit.commit_url, flush=True)
    info = api.model_info(REPO, revision=oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    weights = {name for name in remote if name.endswith(('.gguf', '.safetensors', '.pt'))}
    assert weights == {'reader.safetensors', 'arbiter.pt', 'QwenGram-4B-BF16.gguf',
                       'QwenGram-4B-Q8_0.gguf', 'QwenGram-4B-Q6_K.gguf', 'QwenGram-4B-Q4_K_M.gguf'}
    verified = {}
    for name, spec in {**manifest['files'], 'SHA256.json': {
            'sha256': sha(RELEASE / 'SHA256.json'), 'size': (RELEASE / 'SHA256.json').stat().st_size}}.items():
        entry = remote[name]
        assert entry.size == spec['size'], name
        if entry.lfs:
            assert entry.lfs.sha256 == spec['sha256'], name
            method = 'HF content-addressed LFS SHA256 and byte size'
        else:
            local = hf_hub_download(REPO, name, revision=oid, token=token, local_dir=VERIFY)
            assert sha(local) == spec['sha256'], name
            method = 'Downloaded content SHA256 and byte size'
        verified[name] = {**spec, 'verification': method}
    public = HfApi(token=False).model_info(REPO, revision=oid)
    assert not public.private
    result = {'repo': REPO, 'url': 'https://huggingface.co/' + REPO,
              'commit': oid,
              'public': True, 'verified_files': verified}
    (HERE / 'publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified', len(verified), 'published files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--verify-commit')
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit('Log in to Hugging Face or set HF_TOKEN before publishing')
    try:
        assert not (args.upload and args.verify_commit)
        main(token, args.upload, args.verify_commit)
    except Exception as error:
        raise SystemExit(str(error).replace(token, '[REDACTED]')) from None
