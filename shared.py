from pathlib import Path


def get_unique_filepath(filepath: str | Path) -> Path:
    filepath = Path(filepath)
    parent = filepath.parent
    stem = filepath.stem
    suffix = filepath.suffix

    counter = 1
    new_path = filepath
    while new_path.exists():
        new_path = parent / f"{stem}_{counter}{suffix}"
        counter += 1

    return new_path
