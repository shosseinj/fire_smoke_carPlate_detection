from __future__ import annotations

"""Unit tests for _parse_zip_entry_personnel helper used in ZIP upload."""

from app.core.personnel_store import _parse_zip_entry_personnel


def test_folder_per_person_structure() -> None:
    """Folder name is used as national_code when file is inside a directory."""
    fname, lname, nc = _parse_zip_entry_personnel("1234567890/photo.jpg")
    assert fname == "Unknown"
    assert lname == "Unknown"
    assert nc == "1234567890"


def test_folder_per_person_deeply_nested() -> None:
    """For deeply nested paths, the first path component is used as national_code
    (after normalization, non-digit characters are stripped)."""
    fname, lname, nc = _parse_zip_entry_personnel("root/1234567890/images/photo.jpg")
    assert fname == "Unknown"
    assert lname == "Unknown"
    # parts = ('root', '1234567890', 'images', 'photo.jpg') → parts[0] = 'root'
    # normalize_national_code('root') → '' (no digits)
    assert nc == ""


def test_underscore_filename_pattern() -> None:
    """Underscore-separated filename: fname_lname_nationalCode.ext."""
    fname, lname, nc = _parse_zip_entry_personnel("John_Doe_1234567890.jpg")
    assert fname == "John"
    assert lname == "Doe"
    assert nc == "1234567890"


def test_underscore_filename_multiword_lname() -> None:
    """Multi-word last names joined with underscores."""
    fname, lname, nc = _parse_zip_entry_personnel("Sara_Mohammadi_Tehrani_1234567890.png")
    assert fname == "Sara"
    assert lname == "Mohammadi_Tehrani"
    assert nc == "1234567890"


def test_bare_numeric_stem() -> None:
    """Filename stem that is a valid national_code."""
    fname, lname, nc = _parse_zip_entry_personnel("1234567890.jpg")
    assert fname == "Unknown"
    assert lname == "Unknown"
    assert nc == "1234567890"


def test_non_numeric_filename_returns_empty() -> None:
    """Non-numeric filename with no folder or underscore pattern returns empty nc."""
    fname, lname, nc = _parse_zip_entry_personnel("photo.jpg")
    assert fname == "Unknown"
    assert lname == "Unknown"
    assert nc == ""


def test_folder_takes_priority_over_filename() -> None:
    """When file is in a folder, folder name is used even if filename has underscore pattern."""
    fname, lname, nc = _parse_zip_entry_personnel("1234567890/John_Doe_0000000000.jpg")
    assert fname == "Unknown"
    assert lname == "Unknown"
    assert nc == "1234567890"


def test_persian_digits_in_folder_name() -> None:
    """Persian digits in folder name are normalized to ASCII."""
    fname, lname, nc = _parse_zip_entry_personnel("۱۲۳۴۵۶۷۸۹۰/photo.jpg")
    assert nc == "1234567890"


def test_persian_digits_in_filename() -> None:
    """Persian digits in filename are normalized to ASCII."""
    fname, lname, nc = _parse_zip_entry_personnel("۱۲۳۴۵۶۷۸۹۰.jpg")
    assert nc == "1234567890"
