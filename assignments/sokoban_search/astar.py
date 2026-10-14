"""A* search.

Same signature and conventions as bfs_search (see search.py and bfs.py):
    astar_search(game: Sokoban, node_limit: int) -> SearchResult

`box_distance_heuristic` is a function h(game: Sokoban, state: Objects) which calculates the Manhattan distance to the
nearest goal. Order the frontier by f = g + h, where g is node.path_cost.

TODO:
    Implement an A* search algorithm using the given templates.

Hints:
  * heapq is the standard-library priority queue. Heap entries are compared element by element, so add a tie-breaker
    counter (e.g. (f, counter, node)), Node objects cannot be compared.
  * A state can be reached again by a cheaper path. Keep the best path cost you have found for each state, and only
    (re)add or expand a node if it improves on that.
  * Follow the counting convention in search.py exactly.
  * Return SearchResult(None, nodes_expanded) if the node limit is hit or the frontier empties."""

import heapq  # noqa: F401

from .game.search import Node, SearchResult, reconstruct_plan  # noqa: F401
from .game.sokoban import Objects, Sokoban


def box_distance_heuristic(game: Sokoban, state: Objects) -> int:
    """Sum over boxes of the Manhattan distance to the nearest goal.

    Every push moves one box one square, and every move costs 1."""

    total = 0
    for box_pos in state.boxes:
        total += min(abs(box_pos.x - goal_pos.x) + abs(box_pos.y - goal_pos.y) for goal_pos in game.goals())
    return total


def astar_search(game: Sokoban, node_limit: int):
    raise NotImplementedError("TODO: Implement an A* algorithm")
