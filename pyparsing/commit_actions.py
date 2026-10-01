"""
commit_actions.py

Support for *commit actions*: parse actions whose execution is deferred until
it is certain that the entire parse run succeeded (including any
``parse_all`` check).

While a parse is in progress, every matching expression that has commit
actions contributes a :class:`_CommitEntry` to a *speculative frame*. Frames
are pushed around alternatives (``MatchFirst``/``Or``), optional elements,
repetition attempts, lookaheads and sequences:

- when the guarded parse succeeds, the frame's entries are merged into the
  surrounding frame;
- when the guarded parse fails, the frame (and every entry collected by
  failed candidate matches that share a prefix with the winning path) is
  discarded.

Packrat cache hits do not create new entries: the entries created during the
original (miss) evaluation are replayed instead, and each entry is committed
exactly once even if a memoized sub-expression is reached several times.

Only the top-level frame of a ``parse_string``/``scan_string`` invocation is
ever committed.  If a commit action raises, the already-executed actions
cannot be undone - the side effects are not transactional.
"""
from __future__ import annotations

import threading
import typing

from .exceptions import CommitActionException, ParseBaseException
from .results import ParseResults


class _CommitEntry:
    """A single deferred commit action captured at parse time."""

    __slots__ = ("fn", "instring", "loc", "tokens")

    def __init__(
        self,
        fn: typing.Callable,
        instring: str,
        loc: int,
        tokens: ParseResults,
    ) -> None:
        self.fn = fn
        self.instring = instring
        self.loc = loc
        self.tokens = tokens


class _CommitState:
    """Per-parse-run stack of speculative frames."""

    __slots__ = ("frames", "committed", "previous")

    def __init__(self) -> None:
        self.frames: list[list[_CommitEntry]] = []
        # dedup set so a packrat hit can never commit the same entry twice
        self.committed: set[_CommitEntry] = set()
        # enclosing parse run, when one parse is driven from inside an action
        self.previous: typing.Optional["_CommitState"] = None


_state_store = threading.local()


def _state() -> typing.Optional[_CommitState]:
    return getattr(_state_store, "state", None)


def begin_parse() -> _CommitState:
    """Start a new top-level parse run and its root frame."""
    state = _CommitState()
    state.previous = getattr(_state_store, "state", None)
    _state_store.state = state
    state.frames.append([])
    return state


def end_parse_commit(state: _CommitState) -> None:
    """Commit the root frame after the whole parse (and parse_all) succeeded."""
    previous = state.previous
    frames = state.frames
    if len(frames) != 1:
        # should never happen; defensive cleanup rather than silently dropping
        root = []
        while frames:
            root.extend(frames.pop())
    else:
        root = frames.pop()

    committed = state.committed
    try:
        for entry in root:
            if entry in committed:
                continue
            committed.add(entry)
            try:
                entry.fn(entry.instring, entry.loc, entry.tokens)
            except CommitActionException:
                raise
            except ParseBaseException:
                raise
            except Exception as err:
                raise CommitActionException(
                    entry.instring,
                    entry.loc,
                    "exception raised in commit action; previously executed "
                    "commit actions cannot be rolled back automatically",
                ) from err
    finally:
        _state_store.state = previous


def end_parse_discard(state: _CommitState) -> None:
    """Throw away every collected entry because the parse run failed."""
    _state_store.state = state.previous
    state.frames.clear()


def push_frame() -> None:
    """Push a new speculative frame (entries belong to a candidate match)."""
    state = _state()
    if state is not None:
        state.frames.append([])


def pop_frame(keep: bool) -> typing.List[_CommitEntry]:
    """
    Pop the current speculative frame.

    ``keep=True`` merges its entries into the surrounding frame (the candidate
    is part of the accepted parse). ``keep=False`` discards them (the
    candidate failed or was a lookahead). The frame's entries are returned in
    either case, so that a packrat miss can record them in its cached value.
    """
    state = _state()
    if state is None:
        return []
    frame = state.frames.pop()
    if keep and state.frames:
        state.frames[-1].extend(frame)
    return frame


def replay(entries: typing.Iterable[_CommitEntry]) -> None:
    """Re-attach entries captured by a packrat miss to the current frame."""
    state = _state()
    if state is not None and state.frames:
        state.frames[-1].extend(entries)


def enqueue(entry: _CommitEntry) -> None:
    """
    Record a matched expression's commit entry.

    Without an active parse run (a bare ``expr._parse(...)`` call) the action
    runs immediately, matching how ordinary parse actions behave.
    """
    state = _state()
    if state is None:
        entry.fn(entry.instring, entry.loc, entry.tokens)
    elif state.frames:
        state.frames[-1].append(entry)
