import pytest

from ..dfs import dfs_search
from ..game.search import SearchResult
from .conftest import execute_plan, make_game

LIMIT = 500_000


@pytest.mark.parametrize("name", ["L1", "L2", "L3", "L4"])
def test_finds_a_valid_plan(levels, name):
    result = dfs_search(make_game(levels[name]), LIMIT)
    assert isinstance(result, SearchResult)
    assert result.plan is not None, "DFS should find a solution here"
    assert execute_plan(levels[name], result.plan), "plan does not solve the puzzle"
    assert 0 < result.nodes_expanded <= LIMIT


def test_valid_plan_on_detour_level(levels):
    result = dfs_search(make_game(levels["L5"]), LIMIT)
    assert result.plan is not None and execute_plan(levels["L5"], result.plan)


def test_unsolvable_level_terminates_with_none(levels):
    result = dfs_search(make_game(levels["L6"]), LIMIT)
    assert result.plan is None
    assert 0 < result.nodes_expanded < LIMIT  # emptied the frontier, not the budget


def test_respects_node_limit(levels):
    result = dfs_search(make_game(levels["L4"]), 100)
    assert result.plan is None
    assert result.nodes_expanded <= 100
