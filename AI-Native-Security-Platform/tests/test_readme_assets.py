"""README local image paths exist (case-sensitive) and organizer logos are identified."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
ASSETS = REPO / "assets" / "organizers"

EXPECTED = ("cmc.png", "tsinghua_shenzhen.png", "steinbeis.png")


def test_readme_exists():
    assert README.is_file()


def test_local_readme_images_exist_with_exact_case():
    text = README.read_text(encoding="utf-8")
    srcs = re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
    local = [s for s in srcs if not s.startswith("http://") and not s.startswith("https://")]
    assert local, "README should reference local organizer images"
    missing = []
    case_mismatch = []
    for rel in local:
        p = REPO / rel
        if p.is_file():
            on_disk = {c.name: c for c in p.parent.iterdir() if c.is_file()}
            if p.name not in on_disk:
                case_mismatch.append(rel)
        else:
            missing.append(rel)
    assert not missing, f"missing images: {missing}"
    assert not case_mismatch, f"case mismatch: {case_mismatch}"


def test_organizer_logos_present_and_not_author_affiliation():
    text = README.read_text(encoding="utf-8")
    assert "Authors are affiliated with HUFLIT" in text
    assert "not CMC University" in text
    for name in EXPECTED:
        p = ASSETS / name
        assert p.is_file(), name
        data = p.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert p.stat().st_size > 1000


def test_organizer_logo_order_and_same_row():
    text = README.read_text(encoding="utf-8")
    local = re.findall(r'src="(assets/organizers/[^"]+)"', text)
    assert local[:3] == [
        "assets/organizers/cmc.png",
        "assets/organizers/tsinghua_shenzhen.png",
        "assets/organizers/steinbeis.png",
    ]
    assert "<table>" in text
    assert text.find("cmc.png") < text.find("tsinghua_shenzhen.png") < text.find("steinbeis.png")


def test_root_originals_preserved():
    """User uncommitted root copies must not be deleted."""
    for name in EXPECTED:
        assert (REPO / name).is_file()
