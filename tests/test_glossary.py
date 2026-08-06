"""Integrity tests for the deterministic glossary (data/glossary.py)."""

from gpt_investor.data.glossary import GLOSSARY, GROUPS, define


def test_every_grouped_term_is_defined():
    for _group, terms in GROUPS:
        for term in terms:
            assert term in GLOSSARY, f"{term} listed in GROUPS but missing from GLOSSARY"


def test_every_glossary_term_appears_in_a_group():
    grouped = {t for _g, terms in GROUPS for t in terms}
    for term in GLOSSARY:
        assert term in grouped, f"{term} defined but not shown in any group"


def test_entries_have_definition_and_http_url():
    for term, entry in GLOSSARY.items():
        assert entry["definition"].strip(), f"{term} has an empty definition"
        assert entry["url"].startswith("http"), f"{term} has a non-http url"


def test_define_returns_definition_or_empty():
    assert define("VIX") == GLOSSARY["VIX"]["definition"]
    assert define("not-a-real-term") == ""


def test_no_duplicate_terms_across_groups():
    seen = []
    for _group, terms in GROUPS:
        seen.extend(terms)
    assert len(seen) == len(set(seen)), "a term is listed in more than one group"
