import os
from pathlib import Path

_INVALID_DIR_CHARS: frozenset[str] = frozenset('<>:"/\\|?*\0')


def _validate_file_exists(path: Path, label: str) -> list[str]:
    """Validate that a path exists, is a file, and is writable.

    Args:
        path: Path to validate.
        label: Label used in error messages.

    Returns:
        List of validation error messages.
    """

    errors: list[str] = []
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")
        return errors
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
        return errors
    return errors


def _validate_writable_file(path: Path, label: str) -> list[str]:
    """Validate that a path is writable or can be created in its directory.

    Args:
        path: Path to validate.
        label: Label used in error messages.

    Returns:
        List of validation error messages.
    """

    errors: list[str] = []
    if not path.exists():
        parent_dir: Path = path.parent
        if not parent_dir.exists():
            errors.append(f"{label} directory does not exist: {parent_dir}")
            return errors
        if not parent_dir.is_dir():
            errors.append(f"{label} directory is not a directory: {parent_dir}")
        return errors
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
        return errors
    if not os.access(path, os.W_OK):
        errors.append(f"{label} is not writable: {path}")
    return errors


def _validate_directory_name(name: str) -> list[str]:
    """Validate that a directory name is usable on common filesystems.

    Args:
        name: Directory name to validate.

    Returns:
        List of validation error messages.
    """

    errors: list[str] = []
    if name in {"", ".", ".."}:
        errors.append("subreddit_name must not be empty or '.' or '..'")
        return errors

    invalid_chars: set[str] = {char for char in name if char in _INVALID_DIR_CHARS}
    if invalid_chars:
        invalid_list: str = "".join(sorted(invalid_chars))
        errors.append(f'subreddit_name contains invalid characters ({invalid_list}); avoid <>:"/\\\\|?* and null.')
    return errors


def _validate_empty_directory(target_dir: Path) -> list[str]:
    """Validate that the subreddit directory is empty or missing.

    Args:
        target_dir: Directory path to validate.

    Returns:
        List of validation error messages.
    """

    errors: list[str] = []
    if target_dir.exists():
        if not target_dir.is_dir():
            errors.append(f"subreddit_name is not a directory: {target_dir}")
            return errors
        if any(target_dir.iterdir()):
            errors.append(f"subreddit_name directory already exists and is not empty: {target_dir}")
    return errors
