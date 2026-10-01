#
# patching.py
#
# Patch-planning API: scan a string for expression matches and build a
# reviewable plan of patches (original span, original text, replacement
# value) that callers can selectively apply, instead of blindly rewriting
# the whole string as transform_string() does.
#
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union

from .results import ParseResults
from .util import _flatten

__all__ = [
    "Patch",
    "PatchPlan",
    "PatchPlanError",
    "PatchPlanSourceMismatch",
    "OverlappingPatchError",
    "create_patch_plan",
]


class PatchPlanError(Exception):
    """Base class for exceptions raised by the patch-planning API."""


class PatchPlanSourceMismatch(PatchPlanError):
    """Raised when applying a :class:`PatchPlan` to a string other than the
    one the plan was created from."""


class OverlappingPatchError(PatchPlanError):
    """Raised when a :class:`PatchPlan` contains nested or overlapping
    patches and the plan's overlap policy is ``"reject"``."""


@dataclass(frozen=True)
class Patch:
    """
    A single planned replacement: the half-open span ``[start, end)`` of the
    original source string, the ``original`` text found at that span, and the
    ``replacement`` text produced by the expression's parse actions.
    """

    start: int
    end: int
    original: str
    replacement: str

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid patch span [{self.start}, {self.end})")


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _tab_expansion_map(source: str, tabsize: int = 8) -> list:
    """
    Build a map from character offsets in ``source.expandtabs(tabsize)``
    back to character offsets in ``source``.  The returned list has one
    entry per offset in the expanded string, plus a final entry holding
    ``len(source)``.  Offsets that fall inside a tab expansion map to the
    offset of the ``"\\t"`` character itself.
    """
    mapping = []
    col = 0
    for i, ch in enumerate(source):
        if ch == "\t":
            width = tabsize - (col % tabsize)
            mapping.extend([i] * width)
            col += width
        else:
            mapping.append(i)
            col = 0 if ch in "\r\n" else col + 1
    mapping.append(len(source))
    return mapping


def _map_expanded_span(mapping: Sequence[int], start: int, end: int) -> tuple:
    """
    Map a ``[start, end)`` span in tab-expanded coordinates back to source
    coordinates.  A span boundary falling inside a tab expansion is rounded
    outward, so the whole ``"\\t"`` character is included in the span.
    """
    orig_start = mapping[start]
    orig_end = mapping[end]
    if 0 < end < len(mapping) - 1 and mapping[end] == mapping[end - 1]:
        # end boundary falls inside a tab expansion - include the whole tab
        orig_end += 1
    return orig_start, orig_end


def _tokens_to_replacement(tokens) -> str:
    if isinstance(tokens, ParseResults):
        items = tokens.as_list()
    elif isinstance(tokens, str):
        items = [tokens]
    elif isinstance(tokens, Iterable):
        items = list(tokens)
    else:
        items = [tokens]
    return "".join(str(item) for item in _flatten(items))


def _resolve_overlaps(patches: list, policy: str) -> list:
    ordered = sorted(patches, key=lambda p: (p.start, p.end))
    if policy == "reject":
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start < prev.end:
                raise OverlappingPatchError(
                    "overlapping patches: "
                    f"[{prev.start}, {prev.end}) and [{nxt.start}, {nxt.end})"
                    " (overlap policy is 'reject')"
                )
        return ordered
    kept: list = []
    for patch in ordered:
        if kept and patch.start < kept[-1].end:
            if policy == "last":
                kept[-1] = patch
            # policy == "first": drop the later overlapping patch
        else:
            kept.append(patch)
    return kept


class PatchPlan:
    """
    A reviewable set of :class:`Patch` replacements planned against a
    specific source string.  The plan is bound to the source by a SHA-256
    digest, so it cannot be accidentally applied to a different string.

    Overlapping or nested patches are handled according to the plan's
    ``overlap`` policy:

    - ``"reject"`` (default): raise :class:`OverlappingPatchError`
    - ``"first"``: keep the earliest patch of each overlapping group
    - ``"last"``: keep the latest patch of each overlapping group
    """

    OVERLAP_POLICIES = ("reject", "first", "last")

    def __init__(
        self,
        source: str,
        patches: Iterable[Patch] = (),
        *,
        overlap: str = "reject",
    ):
        if overlap not in self.OVERLAP_POLICIES:
            raise ValueError(
                f"overlap policy must be one of {self.OVERLAP_POLICIES},"
                f" not {overlap!r}"
            )
        self._source = str(source)
        self._digest = _source_digest(self._source)
        self._overlap = overlap
        self._patches: tuple = tuple(_resolve_overlaps(list(patches), overlap))

    @property
    def source(self) -> str:
        """The source string this plan was created from."""
        return self._source

    @property
    def source_digest(self) -> str:
        """SHA-256 hex digest of the source string."""
        return self._digest

    @property
    def overlap_policy(self) -> str:
        return self._overlap

    @property
    def patches(self) -> tuple:
        return self._patches

    def __len__(self) -> int:
        return len(self._patches)

    def __iter__(self) -> Iterator[Patch]:
        return iter(self._patches)

    def __getitem__(self, item) -> Patch:
        return self._patches[item]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self._patches)} patches,"
            f" digest={self._digest[:12]}...)"
        )

    def verify_source(self, text: str) -> bool:
        """Return True if ``text`` is the source this plan was built from."""
        return _source_digest(str(text)) == self._digest

    def _select(self, selected) -> list:
        if selected is None:
            return list(self._patches)
        if callable(selected):
            return [p for p in self._patches if selected(p)]
        chosen = []
        for item in selected:
            if isinstance(item, int):
                chosen.append(self._patches[item])
            elif isinstance(item, Patch):
                if item not in self._patches:
                    raise ValueError(
                        f"patch {item!r} does not belong to this plan"
                    )
                chosen.append(item)
            else:
                raise TypeError(
                    "selected items must be patch indices or Patch objects,"
                    f" not {type(item).__name__}"
                )
        # de-duplicate, preserving plan order
        chosen_ids = {id(p) for p in chosen}
        return [p for p in self._patches if id(p) in chosen_ids]

    def apply(
        self,
        selected: Optional[
            Union[Iterable[Union[int, Patch]], Callable[[Patch], bool]]
        ] = None,
    ) -> str:
        """
        Apply the selected patches to the plan's source string and return the
        patched result.  ``selected`` may be:

        - ``None`` (default): apply all patches
        - an iterable of patch indices or :class:`Patch` objects
        - a callable taking a :class:`Patch` and returning bool

        Unselected matches are left unchanged.
        """
        return self.apply_to(self._source, selected)

    def apply_to(self, text: str, selected=None) -> str:
        """
        Apply the selected patches to ``text``, which must be identical to
        the source string this plan was created from; otherwise raises
        :class:`PatchPlanSourceMismatch`.
        """
        text = str(text)
        if not self.verify_source(text):
            raise PatchPlanSourceMismatch(
                "patch plan was created from a different source string"
                f" (expected digest {self._digest[:12]}..., got"
                f" {_source_digest(text)[:12]}...)"
            )
        chosen = _resolve_overlaps(self._select(selected), "reject")
        out = []
        last = 0
        for patch in chosen:
            out.append(text[last : patch.start])
            out.append(patch.replacement)
            last = patch.end
        out.append(text[last:])
        return "".join(out)


def create_patch_plan(
    expr,
    instring: str,
    max_matches: int = 2**31 - 1,
    *,
    overlap: str = "reject",
    debug: bool = False,
) -> PatchPlan:
    """
    Scan ``instring`` for matches of ``expr`` and build a :class:`PatchPlan`
    describing the replacements that ``expr.transform_string(instring)``
    would perform.  Each :class:`Patch` reports its span as offsets into the
    original ``instring`` - even when tabs are expanded for parsing, the
    expanded positions are mapped back to the original string, so tabs,
    ignored comments, and line endings (including CRLF) outside the matched
    spans are preserved exactly when the plan is applied.

    ``overlap`` selects the policy for nested or overlapping patches:
    ``"reject"`` (default), ``"first"``, or ``"last"``.
    """
    instring = str(instring)
    if expr.keepTabs:
        scan_text = instring
        index_map = None
    else:
        scan_text = instring.expandtabs()
        index_map = _tab_expansion_map(instring)

    patches = []
    for tokens, start, end in expr.scan_string(
        scan_text, max_matches=max_matches, debug=debug
    ):
        if index_map is not None:
            start, end = _map_expanded_span(index_map, start, end)
        patches.append(
            Patch(
                start=start,
                end=end,
                original=instring[start:end],
                replacement=_tokens_to_replacement(tokens),
            )
        )
    return PatchPlan(instring, patches, overlap=overlap)
