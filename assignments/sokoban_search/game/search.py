"""Shared search helpers.

Every search algorithm in this assignment has the same signature:

    def some_search(game: Sokoban, node_limit: int) -> SearchResult

* `game` a Sokoban instance used as your agent's model of the world.
* `node_limit`  the maximum number of nodes the algorithm may expand.

A *state* is an `Objects` value (player position and box positions). It is hashable, so it can go straight into a set or dict.

Counting convention (all algorithms follow it so results are comparable):
* A node is *expanded* when it is removed from the frontier, its state is goal-tested, and (if it is not a goal) its
    children are generated.
* `nodes_expanded` counts every such node, including the final goal node.
* The goal test happens when a node is removed from the frontier.
* Every action costs 1, so a child's `path_cost` is its parent's plus 1.
* If `nodes_expanded` reaches `node_limit` before a solution is found, stop and return a SearchResult with plan=None.
* An example search can be found in `bfs_search`."""

from dataclasses import dataclass

from .sokoban import Action, Objects


@dataclass(slots=True)
class Node:
    """A node in the search tree."""

    state: Objects
    parent: "Node | None" = None
    action: Action | None = None
    path_cost: int = 0


@dataclass(slots=True)
class SearchResult:
    plan: list[Action] | None  # A list of actions, or None if no plan was found
    nodes_expanded: int


def reconstruct_plan(node: Node) -> list[Action]:
    """Follow parent pointers back to the root; return the list of actions."""

    actions: list[Action] = []
    while node.parent is not None:
        if node.action is None:
            raise ValueError("An action in a plan cannot be none")

        actions.append(node.action)
        node = node.parent

    actions.reverse()
    return actions
