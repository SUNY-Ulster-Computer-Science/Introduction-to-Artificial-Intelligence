import pytest

from assignments.sokoban_search.game.search import SearchResult

from ..astar import astar_search
from ..bfs import bfs_search
from .conftest import execute_plan, make_game

LIMIT = 500_000


@pytest.mark.parametrize("name", ["L1", "L2", "L3", "L4", "L5"])
def test_plans_are_optimal(levels, name):
    game = make_game(levels[name])
    result = astar_search(game, LIMIT)
    assert isinstance(result, SearchResult)
    assert result.plan is not None
    assert execute_plan(levels[name], result.plan), "plan does not solve the puzzle"
    optimal = bfs_search(game, LIMIT)
    assert optimal.plan is not None
    assert len(result.plan) == len(optimal.plan), "plan is not optimal"


@pytest.mark.parametrize("name", ["L3", "L4"])
def test_expands_fewer_nodes_than_bfs(levels, name):
    game = make_game(levels[name])
    a = astar_search(game, LIMIT)
    b = bfs_search(game, LIMIT)
    assert a.nodes_expanded < b.nodes_expanded


def test_unsolvable_level_terminates_with_none(levels):
    result = astar_search(make_game(levels["L6"]), LIMIT)
    assert result.plan is None
    assert 0 < result.nodes_expanded < LIMIT


def test_respects_node_limit(levels):
    result = astar_search(make_game(levels["L4"]), 100)
    assert result.plan is None
    assert result.nodes_expanded <= 100
