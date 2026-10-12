from ..bfs import bfs_search
from .conftest import execute_plan, make_game


def test_bfs_solves_and_plan_replays(levels):
    for name in ("L1", "L2", "L3"):
        r = bfs_search(make_game(levels[name]), 200_000)
        assert r.plan is not None and execute_plan(levels[name], r.plan), name


def test_bfs_unsolvable_returns_none(levels):
    r = bfs_search(make_game(levels["L6"]), 200_000)
    assert r.plan is None and r.nodes_expanded < 200_000
