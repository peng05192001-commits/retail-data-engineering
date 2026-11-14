"""Run privacy-safe tests with no external input package."""
from pathlib import Path
import ast
import subprocess
import sys

import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(root)))
    for path in root.rglob('*.ipynb'):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if any(cell.cell_type == 'code' and (cell.outputs or cell.execution_count is not None) for cell in notebook.cells):
            raise ValueError('Public notebooks must not contain saved private outputs: ' + path.name)
    commands = [
        (root / 'integration', [sys.executable, 'scripts/test_public.py']),
        (root / 'cleansing', [sys.executable, '-m', 'unittest', 'test_monetary.MonetaryUnitTests', '-v']),
    ]
    for directory, command in commands:
        subprocess.run(command, cwd=directory, check=True)
    print('Public checks passed: source syntax, notebook structure/output hygiene and data-free regression tests.')


if __name__ == '__main__':
    main()
