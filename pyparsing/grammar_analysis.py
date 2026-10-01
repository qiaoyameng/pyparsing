# grammar_analysis.py
#
# static validation of pyparsing grammars: detect recursive Forward
# cycles that cannot make forward progress, distinguish them from legal
# left recursion and from ordinary terminating recursion, and report
# custom elements that cannot be analyzed statically as "unknown".
#

import typing
from enum import Enum

from .core import (
    And,
    Each,
    FollowedBy,
    Forward,
    MatchFirst,
    NoMatch,
    NotAny,
    Opt,
    Or,
    ParseElementEnhance,
    ParserElement,
    PrecededBy,
    SkipTo,
    ZeroOrMore,
)

__all__ = [
    "GrammarIssue",
    "GrammarIssueKind",
    "GrammarValidationResult",
    "validate_grammar",
]


class GrammarIssueKind(Enum):
    """
    Category of a problem (or potential problem) found by
    :func:`validate_grammar`:

    - ``NO_PROGRESS_LOOP`` - a cycle of :class:`Forward` references that can
      recurse indefinitely without consuming any input and has no terminating
      alternative. Always an error: parsing raises ``RecursionError`` (or
      ``ParseException`` "Forward recursion without base case" when left
      recursion is enabled).
    - ``LEFT_RECURSION`` - a cycle of :class:`Forward` references reachable
      without consuming input, but with a terminating base case. Legal only
      when :meth:`ParserElement.enable_left_recursion` has been called.
    - ``UNKNOWN`` - analysis was inconclusive because the grammar contains
      custom :class:`ParserElement` subclasses whose matching behavior
      cannot be determined statically. Reported as a warning, not an error.
    """

    NO_PROGRESS_LOOP = "no_progress_loop"
    LEFT_RECURSION = "left_recursion"
    UNKNOWN = "unknown"


def _display_name(e: ParserElement) -> str:
    if e.customName:
        return e.customName
    if e.resultsName:
        return e.resultsName
    s = " ".join(str(e).split())
    if not s or s == "None":
        s = type(e).__name__
    if len(s) > 48:
        s = s[:45] + "..."
    return s


class GrammarIssue(typing.NamedTuple):
    """
    A single finding reported by :func:`validate_grammar`.

    - ``kind`` - a :class:`GrammarIssueKind` value
    - ``chain`` - the tuple of :class:`ParserElement` objects forming the
      detected cycle, first element repeated at the end
    - ``message`` - human-readable description of the finding
    """

    kind: GrammarIssueKind
    chain: tuple
    message: str

    @property
    def chain_str(self) -> str:
        """the cycle rendered as a chain of element names, e.g. ``A -> B -> A``"""
        return " -> ".join(_display_name(e) for e in self.chain)

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.message} [chain: {self.chain_str}]"


class GrammarValidationResult:
    """
    Result returned by :func:`validate_grammar` and
    :meth:`ParserElement.validate_grammar`.

    - ``issues`` - all findings, including ``UNKNOWN`` warnings
    - ``errors`` - findings that will make parsing fail
      (``NO_PROGRESS_LOOP``, plus ``LEFT_RECURSION`` when left recursion
      is not enabled)
    - ``warnings`` - inconclusive findings (``UNKNOWN``)
    - ``ok`` - True if there are no errors
    """

    def __init__(self, issues: typing.Iterable[GrammarIssue] = ()) -> None:
        self.issues: list[GrammarIssue] = list(issues)

    @property
    def errors(self) -> list[GrammarIssue]:
        return [
            i
            for i in self.issues
            if i.kind in (GrammarIssueKind.NO_PROGRESS_LOOP, GrammarIssueKind.LEFT_RECURSION)
        ]

    @property
    def warnings(self) -> list[GrammarIssue]:
        return [i for i in self.issues if i.kind is GrammarIssueKind.UNKNOWN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if not self.issues:
            return "grammar validation passed: no problematic recursion detected"
        return "\n".join(str(issue) for issue in self.issues)


# tri-state logic: True / False / None (unknown)
def _tri_all(values) -> typing.Optional[bool]:
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _tri_any(values) -> typing.Optional[bool]:
    values = list(values)
    if any(v is True for v in values):
        return True
    if any(v is None for v in values):
        return None
    return False


def _is_pyparsing_element(e: ParserElement) -> bool:
    return type(e).__module__.startswith("pyparsing")


def _fixpoint(seed, rule):
    """iterate ``rule`` over a {id: value} map until stable"""
    changed = True
    while changed:
        changed = False
        for key, update in seed.items():
            new_value = update()
            if new_value != rule[key]:
                rule[key] = new_value
                changed = True
    return rule


class _GrammarAnalyzer:
    def __init__(self, root: ParserElement) -> None:
        self.root = root
        self.elements: list[ParserElement] = []
        self.forwards: list[Forward] = []
        self._collect()
        self._nullable = self._compute_nullable()
        self._lead_min = self._compute_lead(assume_unknown_nullable=False)
        self._lead_max = self._compute_lead(assume_unknown_nullable=True)

    def _collect(self) -> None:
        # walk the whole grammar, including ignore() expressions
        seen: set[int] = set()
        stack = [self.root]
        while stack:
            e = stack.pop()
            if id(e) in seen:
                continue
            seen.add(id(e))
            self.elements.append(e)
            stack.extend(e.recurse())
            stack.extend(e.ignoreExprs)
        self.forwards = [e for e in self.elements if isinstance(e, Forward)]

    # -- nullability: can this element match without consuming input? --

    def _nullable_rule(self, e: ParserElement, nullable: dict) -> typing.Optional[bool]:
        if isinstance(e, Forward):
            return False if e.expr is None else nullable[id(e.expr)]
        if isinstance(e, (And, Each)):
            return _tri_all(nullable[id(c)] for c in e.exprs)
        if isinstance(e, (Or, MatchFirst)):
            return _tri_any(nullable[id(c)] for c in e.exprs)
        # always match empty: optionals, zero-or-more, lookarounds, skip-to
        if isinstance(e, (Opt, ZeroOrMore, FollowedBy, NotAny, PrecededBy, SkipTo)):
            return True
        if isinstance(e, ParseElementEnhance):
            return False if e.expr is None else nullable[id(e.expr)]
        if isinstance(e, NoMatch):
            return False
        if _is_pyparsing_element(e):
            # mayReturnEmpty is maintained in _may_return_empty, and is
            # lazily computed for some classes (such as Regex)
            return bool(e.mayReturnEmpty)
        # custom element with unknown behavior
        return None

    def _compute_nullable(self) -> dict:
        nullable = {id(e): False for e in self.elements}
        seeds = {
            id(e): (lambda e=e: self._nullable_rule(e, nullable))
            for e in self.elements
        }
        return _fixpoint(seeds, nullable)

    # -- leading Forwards: reachable without consuming input --

    def _lead_rule(
        self, e: ParserElement, lead: dict, assume_unknown_nullable: bool
    ) -> frozenset:
        out: set[Forward] = set()
        # ignore() expressions are attempted at the current position
        for ign in e.ignoreExprs:
            out |= lead[id(ign)]
        if isinstance(e, Forward):
            out.add(e)
        elif isinstance(e, And):
            for c in e.exprs:
                out |= lead[id(c)]
                n = self._nullable[id(c)]
                if n is False or (n is None and not assume_unknown_nullable):
                    break
        elif isinstance(e, (Or, MatchFirst, Each)):
            for c in e.exprs:
                out |= lead[id(c)]
        elif isinstance(e, ParseElementEnhance):
            # includes Opt, ZeroOrMore, OneOrMore, Group, lookarounds, ...
            if e.expr is not None:
                out |= lead[id(e.expr)]
        return frozenset(out)

    def _compute_lead(self, assume_unknown_nullable: bool) -> dict:
        lead = {id(e): frozenset() for e in self.elements}
        seeds = {
            id(e): (lambda e=e: self._lead_rule(e, lead, assume_unknown_nullable))
            for e in self.elements
        }
        return _fixpoint(seeds, lead)

    # -- productivity: can this element match without using the cycle? --

    def _productive_rule(
        self, e: ParserElement, cycle: set, productive: dict
    ) -> typing.Optional[bool]:
        if isinstance(e, Forward):
            if e in cycle:
                return False
            return False if e.expr is None else productive[id(e.expr)]
        if isinstance(e, (And, Each)):
            return _tri_all(productive[id(c)] for c in e.exprs)
        if isinstance(e, (Or, MatchFirst)):
            return _tri_any(productive[id(c)] for c in e.exprs)
        if isinstance(e, (Opt, ZeroOrMore, NotAny, PrecededBy, SkipTo)):
            return True
        if isinstance(e, ParseElementEnhance):
            # includes FollowedBy: succeeds only if its inner expr succeeds
            return True if e.expr is None else productive[id(e.expr)]
        if isinstance(e, NoMatch):
            return False
        if _is_pyparsing_element(e):
            return True
        # custom element with unknown behavior
        return None

    def _compute_productive(self, cycle: set) -> dict:
        productive = {id(e): False for e in self.elements}
        seeds = {
            id(e): (lambda e=e: self._productive_rule(e, cycle, productive))
            for e in self.elements
        }
        return _fixpoint(seeds, productive)

    # -- cycle detection --

    def _edges(self, lead: dict) -> dict:
        edges = {}
        for f in self.forwards:
            targets = lead[id(f.expr)] if f.expr is not None else frozenset()
            edges[f] = sorted(
                (t for t in targets if isinstance(t, Forward)),
                key=_display_name,
            )
        return edges

    @staticmethod
    def _cyclic_sccs(edges: dict) -> list[list[Forward]]:
        # iterative Tarjan
        index: dict = {}
        lowlink: dict = {}
        on_stack: set = set()
        stack: list = []
        counter = [0]
        components = []
        for root in edges:
            if root in index:
                continue
            work = [(root, 0)]
            while work:
                v, child_i = work[-1]
                if child_i == 0:
                    index[v] = lowlink[v] = counter[0]
                    counter[0] += 1
                    stack.append(v)
                    on_stack.add(v)
                descend = False
                succs = edges.get(v, ())
                i = child_i
                while i < len(succs):
                    w = succs[i]
                    if w not in index:
                        work[-1] = (v, i + 1)
                        work.append((w, 0))
                        descend = True
                        break
                    if w in on_stack:
                        lowlink[v] = min(lowlink[v], index[w])
                    i += 1
                if descend:
                    continue
                if lowlink[v] == index[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    components.append(comp)
                work.pop()
                if work:
                    u = work[-1][0]
                    lowlink[u] = min(lowlink[u], lowlink[v])
        return [
            comp
            for comp in components
            if len(comp) > 1 or comp[0] in edges.get(comp[0], ())
        ]

    @staticmethod
    def _cycle_path(start: Forward, cycle: set, edges: dict) -> list:
        stack = [(start, [start])]
        visited = {start}
        while stack:
            node, path = stack.pop()
            for succ in edges.get(node, ()):
                if succ not in cycle:
                    continue
                if succ is start:
                    return path + [start]
                if succ not in visited:
                    visited.add(succ)
                    stack.append((succ, path + [succ]))
        return [start]

    # -- main analysis --

    def analyze(self) -> GrammarValidationResult:
        issues: list[GrammarIssue] = []

        edges_min = self._edges(self._lead_min)
        edges_max = self._edges(self._lead_max)
        cycles_min = self._cyclic_sccs(edges_min)
        cycles_max = self._cyclic_sccs(edges_max)

        forwards_in_min_cycles = {f for comp in cycles_min for f in comp}

        for comp in cycles_min:
            cycle = set(comp)
            productive = self._compute_productive(cycle)
            viability = {
                f: (False if f.expr is None else productive[id(f.expr)])
                for f in comp
            }
            unviable = [f for f in comp if viability[f] is False]
            uncertain = [f for f in comp if viability[f] is None]
            start = (unviable or uncertain or comp)[0]
            path = self._cycle_path(start, cycle, edges_min)
            chain_str = " -> ".join(_display_name(e) for e in path)

            if unviable:
                issues.append(
                    GrammarIssue(
                        GrammarIssueKind.NO_PROGRESS_LOOP,
                        tuple(path),
                        f"no-progress loop: {chain_str} can recurse indefinitely"
                        " without consuming input and has no terminating"
                        " alternative; parsing raises RecursionError (or"
                        " 'Forward recursion without base case' when left"
                        " recursion is enabled)",
                    )
                )
            elif uncertain:
                names = ", ".join(_display_name(f) for f in uncertain)
                issues.append(
                    GrammarIssue(
                        GrammarIssueKind.UNKNOWN,
                        tuple(path),
                        f"cannot statically determine whether cycle"
                        f" {chain_str} terminates: {names} contains custom"
                        " elements with unknown matching behavior",
                    )
                )
            elif not ParserElement._left_recursion_enabled:
                issues.append(
                    GrammarIssue(
                        GrammarIssueKind.LEFT_RECURSION,
                        tuple(path),
                        f"left-recursive cycle {chain_str} has a terminating"
                        " base case, but is only parseable after calling"
                        " ParserElement.enable_left_recursion()",
                    )
                )
            # else: legal left recursion with left recursion enabled - OK

        # cycles that only appear when unknown-nullable custom elements are
        # assumed nullable: report as unknown, never as errors
        for comp in cycles_max:
            if all(f in forwards_in_min_cycles for f in comp):
                continue
            cycle = set(comp)
            start = comp[0]
            path = self._cycle_path(start, cycle, edges_max)
            chain_str = " -> ".join(_display_name(e) for e in path)
            issues.append(
                GrammarIssue(
                    GrammarIssueKind.UNKNOWN,
                    tuple(path),
                    f"possible no-progress loop {chain_str}: cannot statically"
                    " determine whether custom elements in the cycle consume"
                    " input",
                )
            )

        return GrammarValidationResult(issues)


def validate_grammar(grammar: ParserElement) -> GrammarValidationResult:
    """
    Statically validate a pyparsing grammar for recursive :class:`Forward`
    reference cycles, before running it and hitting a bare ``RecursionError``.

    The analysis distinguishes:

    - *no-progress loops* - cycles that recurse without consuming input and
      have no terminating alternative (reported as errors)
    - *left recursion* - cycles with a terminating base case (reported as
      errors only when :meth:`ParserElement.enable_left_recursion` has not
      been called; grammars with left recursion enabled are not rejected)
    - *terminating recursion* - recursion that consumes input before
      recurring (not reported)

    ``ignore()`` expressions, :class:`Opt`, and :class:`ZeroOrMore` elements
    are all included in the analysis. Custom :class:`ParserElement`
    subclasses whose behavior cannot be determined statically yield
    ``UNKNOWN`` warnings instead of errors.

    Example::

        import pyparsing as pp

        expr = pp.Forward()("expr")
        expr <<= expr + "+" + pp.Word(pp.nums)  # missing base case

        result = pp.validate_grammar(expr)
        if not result.ok:
            print(result)
    """
    return _GrammarAnalyzer(grammar).analyze()
