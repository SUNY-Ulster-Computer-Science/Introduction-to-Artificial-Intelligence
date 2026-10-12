from ..agents import SearchAgent
from ..game.search import SearchResult
from ..game.sokoban import Action, Sokoban, parse_level
from ..harness import ALGORITHMS, play

U, D, R = Action.UP, Action.DOWN, Action.RIGHT


def stub_search(plan, log):
    """A fake search function that records how it was called."""

    def search_fn(game, node_limit):
        log.append((game, node_limit))
        return SearchResult(list(plan) if plan is not None else None, 7)

    return search_fn


def first_percept(levels):
    return parse_level(levels["L1"]).reset()


def test_result_is_none_before_acting():
    agent = SearchAgent(stub_search([R], []), 10)
    assert agent.get_plan()[0] is None


def test_searches_once_and_returns_actions_in_order(levels):
    log = []
    agent = SearchAgent(stub_search([R, U, D], log), 10)
    percept = first_percept(levels)
    assert [agent.act(percept) for _ in range(3)] == [R, U, D]
    assert len(log) == 1


def test_passes_a_model_of_the_world_and_the_node_limit(levels):
    log = []
    agent = SearchAgent(stub_search([R], log), node_limit=123)
    percept = first_percept(levels)
    agent.act(percept)
    game, node_limit = log[0]
    assert isinstance(game, Sokoban)
    assert node_limit == 123


def test_exposes_search_result(levels):
    agent = SearchAgent(stub_search([R], []), 10)
    agent.act(first_percept(levels))
    assert agent.get_plan()[1] == 7


def test_returns_none_when_plan_exhausted(levels):
    agent = SearchAgent(stub_search([R], []), 10)
    percept = first_percept(levels)
    assert agent.act(percept) == R
    assert agent.act(percept) is None
    assert agent.act(percept) is None


def test_returns_none_when_no_plan_found_and_does_not_research(levels):
    log = []
    agent = SearchAgent(stub_search(None, log), 10)
    percept = first_percept(levels)
    assert agent.act(percept) is None
    assert agent.act(percept) is None
    assert len(log) == 1


def test_plans_from_the_first_percept_only(levels):
    # Later percepts must not trigger a new search, even if they differ.
    log = []
    agent = SearchAgent(stub_search([R, R], log), 10)
    game = parse_level(levels["L2"])
    percept = game.reset()
    agent.act(percept)
    percept = game.update(Action.RIGHT)
    agent.act(percept)
    assert len(log) == 1


def test_works_end_to_end_with_provided_bfs(levels):
    for name in ("L1", "L2", "L3"):
        game = parse_level(levels[name])
        agent = SearchAgent(ALGORITHMS["bfs"], 200_000)
        assert play(game, agent), name
