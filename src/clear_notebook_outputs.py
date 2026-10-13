"""Clear Notebook outputs before sharing; does not upload or commit anything."""
import argparse
from pathlib import Path
import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    nbformat.write(notebook, args.notebook)
    print(f'Cleared saved outputs: {args.notebook}')


if __name__ == '__main__':
    main()
