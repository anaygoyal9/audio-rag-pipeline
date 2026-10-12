import json

import pytest

from app.rag.query_parser import (
    Filters,
    build_where,
    matches,
    parse_llm_json,
    rule_based_parse,
)

VOCAB = {
    "genres": ["acoustic", "ambient", "classical", "electronic", "hip-hop", "jazz", "lofi", "pop", "rock"],
    "moods": ["chill", "dark", "energetic", "focused", "happy", "melancholic", "romantic", "uplifting"],
}


def test_rules_pick_genre_and_activity_constraints():
    p = rule_based_parse("Upbeat electronic music for a morning workout", VOCAB)
    assert p.filters.genres == ["electronic"]
    assert p.filters.numeric["energy"] == {"$gte": 0.7}
    assert p.filters.numeric["tempo"] == {"$gte": 115}
    assert p.filters.numeric["valence"] == {"$gte": 0.6}  # "upbeat"
    assert p.source == "rules"


def test_rules_study_and_mood_and_hyphenated_genre():
    p = rule_based_parse("dark hip-hop to study to", VOCAB)
    assert p.filters.genres == ["hip-hop"]
    assert p.filters.moods == ["dark"]
    assert p.filters.numeric["energy"] == {"$lte": 0.6}


def test_rules_ignore_words_outside_the_catalogue_vocabulary():
    p = rule_based_parse("polka for a wedding", VOCAB)
    assert p.filters.is_empty()
    assert p.semantic_text == "polka for a wedding"


def test_rules_do_not_match_inside_words():
    p = rule_based_parse("a populist anthem", VOCAB)  # contains "pop"
    assert p.filters.genres == []


def test_stricter_bound_wins_when_rules_overlap():
    p = rule_based_parse("slow chill relax", VOCAB)  # <=0.5 and <=0.45 energy
    assert p.filters.numeric["energy"]["$lte"] == 0.45


def test_where_clause_shapes():
    assert build_where(Filters()) is None
    assert build_where(Filters(genres=["rock"])) == {"genre": {"$in": ["rock"]}}
    where = build_where(Filters(genres=["rock"], numeric={"energy": {"$gte": 0.7, "$lte": 0.9}}))
    assert where == {"$and": [{"genre": {"$in": ["rock"]}}, {"energy": {"$gte": 0.7}}, {"energy": {"$lte": 0.9}}]}


def test_python_filter_agrees_with_where_semantics():
    f = Filters(genres=["rock"], numeric={"energy": {"$gte": 0.7}})
    assert matches({"genre": "rock", "energy": 0.8}, f)
    assert not matches({"genre": "rock", "energy": 0.5}, f)
    assert not matches({"genre": "pop", "energy": 0.9}, f)
    assert not matches({"genre": "rock"}, f)  # missing field never matches


def test_relaxation_ladder_degrades_and_dedupes():
    full = Filters(["rock"], ["dark"], {"energy": {"$gte": 0.7}})
    levels = [lvl for lvl, _ in full.relaxation_ladder()]
    assert levels == [0, 1, 2, 3]
    only_numeric = Filters(numeric={"energy": {"$gte": 0.7}})
    assert [lvl for lvl, _ in only_numeric.relaxation_ladder()] == [0, 1]  # levels 2/3 collapse into 1
    assert Filters().relaxation_ladder() == [(0, Filters())]


def test_llm_json_is_validated_against_vocabulary_and_clamped():
    raw = "Sure! " + json.dumps(
        {
            "semantic_text": "fast driving beats",
            "genres": ["Electronic", "metalcore", 5],  # unknown + wrong type are dropped, case is normalised
            "moods": ["ENERGETIC"],
            "min_tempo": 9999,  # clamped to 300
            "min_energy": 0.8,
            "max_energy": "high",  # wrong type -> ignored
            "playlist_title": "Drive " * 30,
        }
    )
    p = parse_llm_json(raw, "x", VOCAB)
    assert p.filters.genres == ["electronic"]
    assert p.filters.moods == ["energetic"]
    assert p.filters.numeric == {"tempo": {"$gte": 300.0}, "energy": {"$gte": 0.8}}
    assert len(p.title) <= 60
    assert p.source == "llm"


def test_llm_garbage_raises_so_caller_can_fall_back():
    with pytest.raises(ValueError):
        parse_llm_json("I cannot help with that", "x", VOCAB)
    with pytest.raises(ValueError):
        parse_llm_json("[1, 2, 3]", "x", VOCAB)
