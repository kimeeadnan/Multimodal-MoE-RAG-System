"""Regression tests for rule-based MoE router."""

from __future__ import annotations

import pytest

from m3docrag.routing.moe_router import decide_route, example_routing_cases, route_expert


@pytest.mark.parametrize(
    "case",
    example_routing_cases(),
    ids=lambda c: c["query"][:40],
)
def test_example_routing_cases(case: dict) -> None:
    expert, reason, _ = decide_route(case["query"])
    assert expert == case["expected"], (case["query"], expert, reason)


def test_visual_requires_structural_cue_not_color_alone() -> None:
    # Avoid leading wh-words so we exercise the "no page/figure/table" path.
    expert, reason, _ = decide_route(
        "Is there a primary color theme described in the abstract?"
    )
    assert expert == "text"
    assert reason == "text:default"


def test_keyword_wh_question() -> None:
    expert, reason, _ = decide_route("Who played Mr. Simms?")
    assert expert == "keyword"
    assert reason in ("keyword:wh_question", "keyword:short_entity")


def test_route_expert_matches_decide_route_first_element() -> None:
    q = "Table 2 revenue by quarter"
    assert route_expert(q) == decide_route(q)[0]
