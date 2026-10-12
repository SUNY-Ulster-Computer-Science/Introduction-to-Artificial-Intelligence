"""Breadth-first search.

Notice the pieces every algorithm in this assignment shares:
  1. A frontier of nodes waiting to be expanded,
  2. A set of states we have already reached (so we never loop),
  3. The goal test when a node is taken off the frontier,
  4. Node counting and the node limit,
  5. Returning a SearchResult (plan=None means "no plan found")."""

from collections import deque

from .game.search import Node, SearchResult, reconstruct_plan
from .game.sokoban import Sokoban


def bfs_search(game: Sokoban, node_limit: int) -> SearchResult:
    start = Node(game.objects())
    frontier = deque([start])  # FIFO queue: oldest node comes out first
    reached = {start.state}  # States already placed on the frontier
    nodes_expanded = 0

    while frontier:
        if nodes_expanded >= node_limit:  # out of budget
            return SearchResult(None, nodes_expanded)

        node = frontier.popleft()
        nodes_expanded += 1

        if game.is_goal(node.state):
            return SearchResult(reconstruct_plan(node), nodes_expanded)

        for action in game.legal_actions(node.state):
            child_state = game.result(node.state, action)
            if child_state not in reached:
                reached.add(child_state)
                frontier.append(Node(child_state, node, action, node.path_cost + 1))

    return SearchResult(None, nodes_expanded)  # Frontier emptied: no solution exists
