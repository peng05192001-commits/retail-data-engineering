"""Prepare a private, verified local copy under neutral dataset filenames.

No input record bytes are altered. The original package is opened read-only.
The derived manifest records the original manifest hash and each copied hash.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, destination, stage):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Destination must be new; existing data is never overwritten.')
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            contents = {n: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
    else:
        contents = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    candidates = [n for n in contents if PurePosixPath(n).name.endswith('_manifest.json')]
    if len(candidates) != 1:
        raise ValueError('Provide one original input package with exactly one JSON manifest.')
    manifest_name = candidates[0]
    prefix = str(PurePosixPath(manifest_name).parent)
    prefix = '' if prefix == '.' else prefix + '/'
    original_bytes = contents[manifest_name]
    manifest = json.loads(original_bytes)
    alias = manifest['group_alias']
    entries, copies = [], []
    for entry in manifest['files']:
        old_name = entry.get('path') or entry['relative_path']
        path = PurePosixPath(old_name)
        if path.is_absolute() or '..' in path.parts or '\\' in old_name or ':' in old_name:
            raise ValueError('Unsafe manifest path')
        data = contents[prefix + old_name]
        expected = entry.get('sha256') or entry['sha256_digest']
        if digest(data).lower() != expected.lower() or ('bytes' in entry and len(data) != entry['bytes']):
            raise ValueError('Original manifest verification failed: ' + old_name)
        new_name = old_name.replace(alias, 'Retail')
        entries.append({'path': new_name, 'bytes': len(data), 'sha256': digest(data)})
        copies.append((new_name, data))
    required = {'raw_input/Retail_commerce.json', 'raw_input/Retail_operations.xml'} if stage == 'integration' else {
        'Retail_dirty_data.csv', 'Retail_missing_data.csv', 'Retail_outlier_data.csv', 'resource_contract.json', 'warehouses.csv'}
    if not required <= {n for n, _ in copies}:
        raise ValueError('Package does not match the selected stage.')
    if len(entries) != len({e['path'] for e in entries}):
        raise ValueError('Normalised paths collide.')
    # Finish verification before creating any local copies.
    destination.mkdir(parents=True)
    for name, data in copies:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    derived = {'group_alias': 'Retail', 'audience': 'private-local-processing',
               'original_manifest_sha256': digest(original_bytes), 'files': entries}
    (destination / (stage + '_manifest.json')).write_text(json.dumps(derived, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(entries)} verified files; record bytes unchanged. Keep this directory private.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['integration', 'cleansing'])
    args = parser.parse_args()
    prepare(args.source, args.destination, args.stage)
