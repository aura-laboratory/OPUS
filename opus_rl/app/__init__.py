"""Local browser application."""

from hashlib import sha256
from pathlib import Path

FRONTEND = Path(__file__).parent / "frontend"


def frontend_revision(directory=FRONTEND):
    """Identify the HTML, scripts and styles served together by this checkout."""
    digest = sha256()
    for name in ("index.html", "app.js", "style.css", "field-help.js"):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()[:12]
