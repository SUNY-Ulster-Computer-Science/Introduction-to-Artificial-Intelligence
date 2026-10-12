"""Depth-first search.

Same signature and conventions as bfs_search (see search.py and bfs.py):
    dfs_search(game: Sokoban, node_limit: int) -> SearchResult

Hints:
  * The only big structural difference from BFS is which end of the frontier you take nodes from.
    A Python list works as a stack.
  * Make sure the search terminates on levels with cycles (Sokoban has plenty: you can always walk back and forth).
    Decide what to remember so you never go around in circles.
  * Follow the counting convention in search.py exactly.
  * Return SearchResult(None, nodes_expanded) if the node limit is hit or the frontier empties."""

from .game.search import Node, SearchResult, reconstruct_plan  # noqa: F401
from .game.sokoban import Sokoban


def dfs_search(game: Sokoban, node_limit: int) -> SearchResult:
    raise NotImplementedError("TODO: Implement a DFS algorithm")
