"""
Helper utilities.
"""

import os


def format_file_size(size_bytes: int) -> str:
    """Konversi ukuran file dari byte ke format yang mudah dibaca."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_file_icon(file_type: str) -> str:
    """Emoji icon berdasarkan tipe file."""
    icons = {
        ".pdf": "📕",
        ".docx": "📘",
        ".xlsx": "📗",
        ".csv": "📊",
        ".pptx": "📙",
        ".txt": "📄",
    }
    return icons.get(file_type, "📁")


def is_supported_file(file_path: str, extensions: list) -> bool:
    """Cek apakah file memiliki ekstensi yang didukung."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in extensions
