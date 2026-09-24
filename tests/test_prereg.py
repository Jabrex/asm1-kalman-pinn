"""PREREGISTRATION.md template: every field renders, every section exists, style rules hold."""

from __future__ import annotations

import re

from scripts import write_prereg as wp


def _context():
    names = {a or b for a, b in re.findall(r"\$(?:\{([a-z_][a-z0-9_]*)\}|([a-z_][a-z0-9_]*))", wp.TEMPLATE.read_text(encoding="utf-8"))}
    return {n: "VALUE_%s" % n for n in names}


def test_template_renders_every_field():
    text = wp.render(_context())
    assert "$" not in text and "VALUE_kinetic_subset" in text


def test_sections_and_hypotheses_present():
    text = wp.render(_context())
    for heading in wp.REQUIRED_HEADINGS:
        assert heading in text
    for h in range(1, 8):
        assert "**H%d.**" % h in text


def test_template_follows_the_style_rules():
    assert wp.style_problems(wp.TEMPLATE.read_text(encoding="utf-8")) == []


def test_style_check_flags_problems():
    assert wp.style_problems("A robust result \u2014 here") == ["banned word 'robust'", "em dash"]


def test_template_fields_match_the_context_builder():
    names = {a or b for a, b in re.findall(r"\$(?:\{([a-z_][a-z0-9_]*)\}|([a-z_][a-z0-9_]*))", wp.TEMPLATE.read_text(encoding="utf-8"))}
    assert names == set(wp.CONTEXT_KEYS)
