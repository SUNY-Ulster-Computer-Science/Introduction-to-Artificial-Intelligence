"""An agent maps percepts to actions. The harness calls `act(percept)` once per time step and executes whatever action
comes back."""

from collections.abc import Callable

from .game.search import SearchResult
from .game.sokoban import Action, Percept, Sokoban


class SearchAgent:
    """A goal-based agent that plans with any search function.

    `search_fn` must follow the shared signature from search.py:
        search_fn(game: Sokoban, node_limit: int) -> SearchResult

    TODO:
      * _plan():
          - Build the agent's own private model of the world with Sokoban.from_percept(percept), and call search_fn,
            returning the result.
      * act(percept):
          - On every call, return the next action of the saved plan.
          - When the plan is exhausted, or no plan was found (plan is None), return None.
        The agent must never search more than once per episode."""

    def __init__(self, search_fn: Callable[[Sokoban, int], SearchResult], node_limit: int) -> None:
        self._search_fn = search_fn
        self._node_limit = node_limit
        self._planned_actions: list[Action] | None = None
        self._planned_nodes: int = 0

    def _plan(self, percept: Percept) -> SearchResult:
        raise NotImplementedError("TODO: Call the search function once and save the search result for acting")

    def get_plan(self) -> tuple[list[Action] | None, int]:
        """Returns the planned out actions and the number of nodes expanded."""

        return self._planned_actions, self._planned_nodes

    def act(self, percept: Percept) -> Action | None:
        # Init plan on first action only
        if self._planned_actions is None:
            search_result = self._plan(percept)
            self._planned_actions = search_result.plan
            self._planned_nodes = search_result.nodes_expanded

        raise NotImplementedError("TODO: Implement acting based on a percept")
