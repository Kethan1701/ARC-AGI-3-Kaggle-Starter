"""The model policy, and the guarantee that it can only ever help.

The agent without a model scored 0.26. Everything here exists to make one
claim testable: wiring a 27B model in front of the search cannot score
*less* than that, because every way the model can fail -- absent, out of
memory, wedged, timing out, replying with prose, replying with a number
that means nothing -- lands on `None`, and `None` runs the search that
already scored.

That is a structural claim, not a hope, so it is tested at both ends: the
parser refuses everything that is not a usable index, and `_select` is
driven with a policy that fails in each of those ways and asserted to
come out at `probe-untried`.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_agent_class():
    # my_agent imports `agents.agent` from the vendored framework.
    vendor = str(ROOT / "vendor" / "ARC-AGI-3-Agents")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    spec = importlib.util.spec_from_file_location(
        "agent_policy_under_test", ROOT / "agent" / "my_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    module = _load_agent_class()
    # The engine is a process-wide singleton; a test that installs a fake
    # must not leak it into the next one.
    module._PolicyEngine._instance = None
    yield module
    module._PolicyEngine._instance = None


# ---------------------------------------------------------------------
# Shipping configuration
# ---------------------------------------------------------------------
def test_the_policy_is_off_by_default(mod):
    """With ARC_POLICY unset not one line of model code runs, so the
    default build is the agent that scored 0.26."""
    assert mod.POLICY_ENABLED == 0


def test_eager_mode_is_off_by_default(mod):
    """enforce_eager=True measured 0.4 actions/s against 47.8 with
    CUDA-graph capture -- same card, same model, same batch of 25. It is
    a debugging aid and nothing else."""
    assert mod.POLICY_EAGER == 0


def test_an_agent_built_without_the_policy_holds_no_model(mod):
    agent = mod.MyAgent.__new__(mod.MyAgent)
    agent._act_tried = mod.Counter()
    agent._act_changed = mod.Counter()
    agent._policy = None
    assert agent._policy is None


# ---------------------------------------------------------------------
# Parsing: everything that is not an index is None
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply, expected",
    [
        ("3", 3),
        ("  7  ", 7),
        ("0", 0),
        ("Option 5", 5),
        ("5.", 5),
        ("<think>maybe 99</think>2", 2),
        ("<think>\nlong\nthought\n</think>\n\n11", 11),
        ("<THINK>x</THINK>4", 4),
    ],
)
def test_a_usable_index_is_recovered(mod, reply, expected):
    assert mod.parse_choice(reply, 24) == expected


@pytest.mark.parametrize(
    "reply",
    [
        None,
        "",
        "   ",
        "I cannot help with that",
        "<think>the answer is 3",          # never closed, never answered
        "<think>3</think>",                # thought only
        "24",                              # one past the end
        "99",
        "-1",
        "none",
    ],
)
def test_anything_unusable_is_none(mod, reply):
    assert mod.parse_choice(reply, 24) is None


def test_an_empty_option_list_is_never_indexed(mod):
    assert mod.parse_choice("0", 0) is None


def test_the_think_block_cannot_smuggle_in_an_answer(mod):
    """The number inside the reasoning is not the answer, and taking it
    would mean acting on a thought the model then discarded."""
    assert mod.parse_choice("<think>I'll say 9</think>1", 24) == 1


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
def test_the_grid_renders_one_character_per_cell(mod):
    grid = tuple(tuple((x + y) % 16 for x in range(64)) for y in range(64))
    lines = mod.render_grid(grid).split("\n")
    assert len(lines) == 66                       # two rulers + 64 rows
    body = lines[2:]
    assert all(len(line.split("  ")[-1]) == 64 for line in body)


def test_the_column_rulers_actually_number_the_columns(mod):
    grid = tuple(tuple(0 for _ in range(64)) for _ in range(64))
    tens, ones = mod.render_grid(grid).split("\n")[:2]
    assert tens.strip()[13] == "1" and ones.strip()[13] == "3"   # column 13
    assert tens.strip()[63] == "6" and ones.strip()[63] == "3"   # column 63


def test_volatile_cells_are_shown_as_unknown(mod):
    """Those are the HUD. Rendering a step counter as real content is how
    a frame-differ ends up clicking the scoreboard all game."""
    grid = tuple(tuple(5 for _ in range(4)) for _ in range(2))
    out = mod.render_grid(grid, ignore=frozenset({0, 1}))
    row0 = out.split("\n")[2]
    assert row0.endswith("??55")


def test_an_empty_grid_does_not_explode(mod):
    assert mod.render_grid(()) == "(empty)"
    assert mod.render_grid(((),)) == "(empty)"


def _reconstruct(mod, rendered, height, width):
    """Rebuild the grid from its rendering, honouring row-range labels."""
    out = [None] * height
    for line in rendered.split("\n")[2:]:
        tag, cells = line.rsplit("  ", 1)
        tag = tag.strip()
        if "-" in tag:
            lo, hi = (int(v) for v in tag.split("-"))
        else:
            lo = hi = int(tag)
        for y in range(lo, hi + 1):
            out[y] = cells
    return out


def test_collapsing_identical_rows_is_lossless(mod):
    """A compression that loses coordinates is worse than none: the whole
    point of the rendering is that the model can name a cell."""
    assert mod.POLICY_COLLAPSE_ROWS == 1

    grid = []
    for y in range(64):
        if 20 <= y < 44:
            grid.append(tuple(3 if 10 <= x < 54 else 0 for x in range(64)))
        else:
            grid.append(tuple(0 for _ in range(64)))
    grid[5] = tuple(7 if 5 <= x < 8 else 0 for x in range(64))
    grid = tuple(grid)

    rebuilt = _reconstruct(mod, mod.render_grid(grid), 64, 64)
    assert all(row is not None for row in rebuilt), "a row went missing"
    for y in range(64):
        expected = "".join(mod._HEX[v] for v in grid[y])
        assert rebuilt[y] == expected, f"row {y} did not survive the round trip"


def test_collapsing_actually_shrinks_a_typical_board(mod):
    """If it does not, the prompt does not fit the run's time budget."""
    grid = tuple(
        tuple(3 if (20 <= y < 44 and 10 <= x < 54) else 0 for x in range(64))
        for y in range(64)
    )
    lines = mod.render_grid(grid).split("\n")
    assert len(lines) < 12, f"barely collapsed: {len(lines)} lines"


def test_a_board_with_no_repeats_still_renders_every_row(mod):
    grid = tuple(tuple((x + y) % 16 for x in range(64)) for y in range(64))
    rebuilt = _reconstruct(mod, mod.render_grid(grid), 64, 64)
    assert all(row is not None for row in rebuilt)


def test_a_click_is_described_with_its_coordinates(mod):
    grid = tuple(tuple(3 for _ in range(64)) for _ in range(64))
    code = mod.encode_action(mod._ACTION6_ID, 12, 30)
    text = mod.describe_code(code, grid)
    assert "column 12" in text and "row 30" in text


def test_simple_actions_are_named_but_not_interpreted(mod):
    """Calling ACTION1 'move up' would be a guess -- the mapping differs
    per game and nothing in the frame declares it. A model told a fact we
    do not have will act on it."""
    grid = tuple(tuple(0 for _ in range(8)) for _ in range(8))
    text = mod.describe_code(mod.encode_action(1, None, None), grid)
    assert "ACTION1" in text
    for lie in ("up", "down", "left", "right"):
        assert lie not in text.lower()


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
def _small_grid():
    return tuple(tuple((x * y) % 16 for x in range(8)) for y in range(8))


def test_the_prompt_offers_exactly_the_candidates_given(mod):
    grid = _small_grid()
    codes = [mod.encode_action(i, None, None) for i in range(1, 6)]
    prompt = mod.build_prompt(grid, codes, [], level=1)
    for i in range(len(codes)):
        assert f"  {i}: " in prompt
    assert f"  {len(codes)}: " not in prompt


def test_the_prompt_is_a_pure_function_of_its_arguments(mod):
    """It has to be, or an A/B measures two different prompts."""
    grid = _small_grid()
    codes = [mod.encode_action(1, None, None)]
    history = [(mod.encode_action(2, None, None), True)]
    a = mod.build_prompt(grid, codes, history, level=2)
    b = mod.build_prompt(grid, codes, history, level=2)
    assert a == b


def test_history_reports_only_whether_the_board_moved(mod):
    grid = _small_grid()
    codes = [mod.encode_action(1, None, None)]
    history = [
        (mod.encode_action(2, None, None), True),
        (mod.encode_action(3, None, None), False),
    ]
    prompt = mod.build_prompt(grid, codes, history, level=1)
    assert "the board changed" in prompt
    assert "nothing happened" in prompt


def test_history_is_capped(mod):
    grid = _small_grid()
    codes = [mod.encode_action(1, None, None)]
    history = [(mod.encode_action(2, None, None), True)] * 100
    prompt = mod.build_prompt(grid, codes, history, level=1)
    assert prompt.count("the board changed") == mod.POLICY_HISTORY


# ---------------------------------------------------------------------
# Engine: batching, and never wedging a game thread
# ---------------------------------------------------------------------
class _Out:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text})()]


class _FakeLLM:
    """Records the batch sizes it is handed."""

    def __init__(self, reply="0", delay=0.0, boom=False):
        self.reply = reply
        self.delay = delay
        self.boom = boom
        self.batches: list[int] = []
        self.lock = threading.Lock()

    def generate(self, prompts, sampling, use_tqdm=False):
        with self.lock:
            self.batches.append(len(prompts))
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("CUDA out of memory")
        # Echo the prompt back so the batching test can prove each caller
        # got its own answer and not its neighbour's.
        return [_Out(p) for p in prompts]


def _engine_with(mod, llm):
    engine = mod._PolicyEngine()
    engine._llm = llm
    engine._wrap = lambda text: text
    engine._sampling = None
    worker = threading.Thread(target=engine._serve, daemon=True)
    worker.start()
    return engine


def test_concurrent_callers_each_get_their_own_answer(mod):
    """25 game threads share one engine. Crossing two replies would make
    one game act on another game's board."""
    llm = _FakeLLM()
    engine = _engine_with(mod, llm)

    results: dict[int, str] = {}
    lock = threading.Lock()

    def ask(i):
        reply = engine.submit(f"prompt-{i}")
        with lock:
            results[i] = reply

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == 25
    for i, reply in results.items():
        assert reply == f"prompt-{i}"


def test_concurrent_callers_are_actually_batched(mod):
    """Serving 25 threads one at a time measured 0.4 actions/s against
    47.8 batched. If this regresses to batches of 1, the run does not
    finish."""
    llm = _FakeLLM(delay=0.05)
    engine = _engine_with(mod, llm)

    threads = [
        threading.Thread(target=lambda i=i: engine.submit(f"p{i}"))
        for i in range(25)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert llm.batches, "the worker never ran"
    assert max(llm.batches) > 1, f"never batched: {llm.batches}"


def test_a_failing_batch_releases_every_waiter(mod):
    """The one bug here that takes the whole run down: a thread parked on
    an Event that is never set burns its game for the rest of the run."""
    llm = _FakeLLM(boom=True)
    engine = _engine_with(mod, llm)

    done = []

    def ask(i):
        done.append(engine.submit(f"p{i}"))

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert all(not t.is_alive() for t in threads), "a caller is still parked"
    assert len(done) == 8
    assert all(reply is None for reply in done)


def test_an_engine_that_cannot_load_fails_once_and_stays_failed(mod):
    """Reloading a 28GB model per action on a box with no GPU would cost
    more than the run."""
    engine = mod._PolicyEngine()
    attempts = []

    def explode():
        attempts.append(1)
        raise RuntimeError("no GPU")

    engine._load = explode
    assert engine.submit("x") is None
    assert engine.submit("x") is None
    assert engine.submit("x") is None
    assert len(attempts) == 1
    assert engine.failed


def test_a_load_that_exits_the_process_is_survived(mod):
    """A failed vLLM engine bootstrap can surface as SystemExit from a
    worker process, which is not an Exception."""
    engine = mod._PolicyEngine()

    def explode():
        raise SystemExit(1)

    engine._load = explode
    assert engine.submit("x") is None
    assert engine.failed


def test_a_keyboard_interrupt_is_never_swallowed(mod):
    engine = mod._PolicyEngine()

    def explode():
        raise KeyboardInterrupt

    engine._load = explode
    with pytest.raises(KeyboardInterrupt):
        engine.submit("x")


# ---------------------------------------------------------------------
# VisionPolicy
# ---------------------------------------------------------------------
class _StubEngine:
    def __init__(self, reply, failed=False):
        self.reply = reply
        self.failed = failed
        self.calls = 0

    def submit(self, prompt):
        self.calls += 1
        return self.reply


def _policy_with(mod, engine):
    mod._PolicyEngine._instance = engine
    return mod.VisionPolicy("test-game")


def test_a_good_reply_becomes_an_index(mod):
    policy = _policy_with(mod, _StubEngine("2"))
    codes = [mod.encode_action(i, None, None) for i in range(1, 6)]
    assert policy.rank(_small_grid(), codes, [], 1) == 2
    assert policy.used == 1


def test_a_dead_engine_retires_the_policy_without_asking(mod):
    engine = _StubEngine(None, failed=True)
    policy = _policy_with(mod, engine)
    codes = [mod.encode_action(1, None, None)]
    assert policy.rank(_small_grid(), codes, [], 1) is None
    assert policy.retired
    assert engine.calls == 0


def test_an_empty_candidate_list_is_never_sent(mod):
    engine = _StubEngine("0")
    policy = _policy_with(mod, engine)
    assert policy.rank(_small_grid(), [], [], 1) is None
    assert engine.calls == 0


def test_persistent_refusal_retires_the_policy(mod):
    """A model that cannot help this game should stop being paid to say
    so -- at ~20ms a call across a 5000-action budget, asking forever is
    real time taken from games it could help."""
    engine = _StubEngine("I don't know")
    policy = _policy_with(mod, engine)
    codes = [mod.encode_action(1, None, None)]

    for _ in range(mod.POLICY_GIVE_UP):
        assert policy.rank(_small_grid(), codes, [], 1) is None

    assert policy.retired
    before = engine.calls
    assert policy.rank(_small_grid(), codes, [], 1) is None
    assert engine.calls == before, "kept asking after retiring"


def test_one_good_answer_resets_the_patience(mod):
    class Alternating:
        failed = False

        def __init__(self):
            self.n = 0

        def submit(self, prompt):
            self.n += 1
            return "0" if self.n % 2 == 0 else "no idea"

    policy = _policy_with(mod, Alternating())
    codes = [mod.encode_action(1, None, None)]
    for _ in range(mod.POLICY_GIVE_UP * 2):
        policy.rank(_small_grid(), codes, [], 1)
    assert not policy.retired


def test_an_index_past_the_shown_options_is_refused(mod):
    """The model is shown at most POLICY_CANDIDATES options; a number
    beyond them refers to nothing."""
    policy = _policy_with(mod, _StubEngine(str(mod.POLICY_CANDIDATES + 5)))
    codes = [mod.encode_action(mod._ACTION6_ID, i, 0) for i in range(64)]
    assert policy.rank(_small_grid(), codes, [], 1) is None


def test_the_option_list_is_capped(mod):
    """Showing all ~117 candidates buries the useful ones and costs
    tokens for no signal."""
    captured = {}

    class Capturing:
        failed = False

        def submit(self, prompt):
            captured["prompt"] = prompt
            return "0"

    policy = _policy_with(mod, Capturing())
    codes = [mod.encode_action(mod._ACTION6_ID, i % 64, i // 64) for i in range(117)]
    policy.rank(_small_grid(), codes, [], 1)
    options = re.findall(r"^  (\d+): ", captured["prompt"], re.MULTILINE)
    assert len(options) == mod.POLICY_CANDIDATES
    assert options == [str(i) for i in range(mod.POLICY_CANDIDATES)]


def test_a_prompt_that_cannot_be_built_is_not_fatal(mod):
    """A frame shaped differently than expected -- what a future format
    change looks like -- must cost one action of nothing, not the game."""
    engine = _StubEngine("0")
    policy = _policy_with(mod, engine)
    assert policy.rank("not a grid", [1, 2, 3], [], 1) is None
    assert engine.calls == 0, "sent a prompt it failed to build"


# ---------------------------------------------------------------------
# The decision point: what `_select` does with a policy
# ---------------------------------------------------------------------
class _FixedPolicy:
    """A policy whose answer the test dictates. `None` is every real
    failure mode collapsed into one: no model, no GPU, OOM, timeout,
    prose, an out-of-range number."""

    retired = False

    def __init__(self, answer):
        self.answer = answer
        self.offered: list[list[int]] = []

    def rank(self, grid, codes, history, level, ignore=frozenset()):
        self.offered.append(list(codes))
        return self.answer


def _agent_with_policy(mod, policy):
    agent = mod.MyAgent.__new__(mod.MyAgent)
    agent.game_id = "test"
    agent._act_tried = mod.Counter()
    agent._act_changed = mod.Counter()
    agent._want_cycle = False
    agent._plan = mod.deque()
    agent._last_productive = False
    agent._last_code = None
    agent._momentum = 0
    agent._exploding = False
    agent._edges = {}
    agent._fatal = set()
    agent._deaths = {}
    agent._steps = 0
    agent._stuck_steps = 0
    agent._breadth = mod.BREADTH_INITIAL
    agent._level = 1
    agent._volatile = frozenset()
    agent._policy = policy
    agent._act_history = []

    node = 1
    codes = [mod.encode_action(a, None, None) for a in range(1, 6)] + [
        mod.encode_action(mod._ACTION6_ID, i % 64, i // 64) for i in range(60)
    ]
    agent._candidates = lambda *a, **k: codes
    agent._total = {node: len(codes)}
    agent._cursor = {node: 0}
    agent._route_to_frontier = lambda start: None
    return agent, node, codes


def test_the_policy_choice_becomes_the_action(mod):
    policy = _FixedPolicy(3)
    agent, node, _ = _agent_with_policy(mod, policy)

    code, why = agent._select(node, _small_grid(), object())

    assert why == "policy"
    assert code == policy.offered[0][3]


def test_a_silent_policy_falls_through_to_the_search(mod):
    """This is the floor. Every model failure arrives here as None, and
    what happens next has to be exactly the agent that scored 0.26."""
    agent, node, codes = _agent_with_policy(mod, _FixedPolicy(None))

    code, why = agent._select(node, _small_grid(), object())

    assert why == "probe-untried"
    assert code == codes[0]
    assert agent._cursor[node] == 1


def test_a_retired_policy_is_not_consulted(mod):
    policy = _FixedPolicy(3)
    policy.retired = True
    agent, node, _ = _agent_with_policy(mod, policy)

    _, why = agent._select(node, _small_grid(), object())

    assert why == "probe-untried"
    assert policy.offered == []


def test_the_policy_is_never_offered_a_dead_action(mod):
    """A dead action is one this game has already shown it ignores.
    Offering it back invites the model to spend the budget re-learning
    what the search proved."""
    policy = _FixedPolicy(None)
    agent, node, codes = _agent_with_policy(mod, policy)
    agent._act_tried[1] = mod.DEAD_ACTION_TRIALS
    dead = mod.encode_action(1, None, None)
    assert agent._is_dead(dead)

    agent._select(node, _small_grid(), object())

    assert dead not in policy.offered[0]


def test_the_policy_is_never_offered_a_fatal_action(mod):
    policy = _FixedPolicy(None)
    agent, node, codes = _agent_with_policy(mod, policy)
    agent._fatal.add((node, codes[2]))

    agent._select(node, _small_grid(), object())

    assert codes[2] not in policy.offered[0]


def test_the_policy_is_never_offered_an_action_already_tried_here(mod):
    policy = _FixedPolicy(None)
    agent, node, codes = _agent_with_policy(mod, policy)
    agent._edges[node] = {codes[0]: 99, codes[1]: 98}

    agent._select(node, _small_grid(), object())

    offered = policy.offered[0]
    assert codes[0] not in offered and codes[1] not in offered


def test_the_cursor_does_not_pay_twice_for_what_the_policy_spent(mod):
    """The cursor is not the only thing spending a node's candidates.
    Without the skip it walks back over the policy's picks and buys an
    answer `_edges` already holds -- one wasted billed action each, on a
    quadratic score."""
    agent, node, codes = _agent_with_policy(mod, _FixedPolicy(None))
    agent._edges[node] = {codes[0]: 5, codes[1]: 6}

    code, why = agent._select(node, _small_grid(), object())

    assert why == "probe-untried"
    assert code == codes[2], "re-probed a candidate already in the graph"


def test_without_a_policy_the_cursor_is_unchanged(mod):
    """The skip above is a real behaviour change, so it stays off for the
    no-model agent until it has been measured on its own."""
    assert mod.SKIP_RECORDED == 0

    agent, node, codes = _agent_with_policy(mod, None)
    agent._edges[node] = {codes[0]: 5, codes[1]: 6}

    code, why = agent._select(node, _small_grid(), object())

    assert why == "probe-untried"
    assert code == codes[0], "no-model behaviour drifted from the 0.26 agent"
