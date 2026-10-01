#
# test_commit_actions.py
#
# Tests for deferred commit actions (ParserElement.set_commit_action /
# ParserElement.add_commit_action). Commit actions must only run after the
# complete parse succeeds (including any parse_all check); actions registered
# by parse branches that later fail must be discarded.
#
import pytest

import pyparsing as pp
from pyparsing import (
    FollowedBy,
    Forward,
    Literal,
    MatchFirst,
    Optional,
    Or,
    Word,
    alphas,
    nums,
)


@pytest.fixture(autouse=True)
def reset_memoization():
    yield
    pp.ParserElement.disable_memoization()


def recorder(log, label):
    """Return a commit action that appends (label, matched text) to log."""

    def record(s, loc, toks):
        log.append((label, toks[0]))

    return record


def test_commit_action_runs_after_successful_parse():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))

    result = word.parse_string("abc")

    assert result.as_list() == ["abc"]
    assert log == [("word", "abc")]


def test_commit_action_receives_string_loc_toks():
    seen = []
    word = Word(alphas).add_commit_action(lambda s, loc, toks: seen.append((s, loc, toks[0])))

    word.parse_string("abc")

    assert seen == [("abc", 0, "abc")]


def test_commit_action_not_run_when_parse_fails():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))
    grammar = word + Literal("!")

    with pytest.raises(pp.ParseException):
        grammar.parse_string("abc")

    assert log == []


def test_commit_action_not_run_when_parse_all_finds_trailing_characters():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))

    # the grammar itself matches, but trailing characters make parse_all fail,
    # so the commit action must not run
    with pytest.raises(pp.ParseException):
        word.parse_string("abc123", parse_all=True)

    assert log == []

    # same grammar and input without parse_all: the parse succeeds and the
    # commit action runs
    word.parse_string("abc123")
    assert log == [("word", "abc")]


def test_common_prefix_branches_only_commit_winning_branch():
    log = []
    branch1 = (
        Literal("a").add_commit_action(recorder(log, "b1-a"))
        + Literal("b").add_commit_action(recorder(log, "b1-b"))
    )
    branch2 = (
        Literal("a").add_commit_action(recorder(log, "b2-a"))
        + Literal("c").add_commit_action(recorder(log, "b2-c"))
    )
    grammar = MatchFirst([branch1, branch2])

    # branch1 shares the "a" prefix, registers its commit action, and then
    # fails on "c" - its registered actions must be discarded
    result = grammar.parse_string("ac", parse_all=True)

    assert result.as_list() == ["a", "c"]
    assert log == [("b2-a", "a"), ("b2-c", "c")]


def test_or_with_common_prefix_commits_longest_match_only():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))
    semi = Literal(";").add_commit_action(recorder(log, "semi"))
    grammar = Or([word + semi, word])

    grammar.parse_string("abc;", parse_all=True)

    # Or first probes with do_actions=False, then re-parses the longest
    # alternative - each element's commit action must run exactly once
    assert log == [("word", "abc"), ("semi", ";")]


def test_optional_discards_actions_of_failed_inner_expression():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))
    bang = Literal("!").add_commit_action(recorder(log, "bang"))
    grammar = word[1, ...] + Optional(bang)

    grammar.parse_string("aa bb", parse_all=True)
    assert log == [("word", "aa"), ("word", "bb")]

    log.clear()
    grammar.parse_string("aa !", parse_all=True)
    assert log == [("word", "aa"), ("bang", "!")]


def test_repetition_commits_each_matched_item_once():
    log = []
    item = Word(alphas).add_commit_action(recorder(log, "item"))
    grammar = item[1, ...]

    # the final, failed repetition attempt must not leave stray actions
    grammar.parse_string("aa bb cc", parse_all=True)

    assert log == [("item", "aa"), ("item", "bb"), ("item", "cc")]


def test_commit_actions_run_in_registration_order():
    log = []
    first = Word(alphas).add_commit_action(recorder(log, "first"))
    second = Word(nums).add_commit_action(recorder(log, "second"))
    third = Literal("!").add_commit_action(recorder(log, "third"))

    (first + second + third).parse_string("abc 123 !", parse_all=True)

    assert log == [("first", "abc"), ("second", "123"), ("third", "!")]


def test_multiple_commit_actions_on_one_expression():
    log = []
    word = Word(alphas)
    word.add_commit_action(recorder(log, "one")).add_commit_action(recorder(log, "two"))

    word.parse_string("abc")

    assert log == [("one", "abc"), ("two", "abc")]


def test_set_commit_action_replaces_and_none_clears():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "old"))
    word.set_commit_action(recorder(log, "new"))

    word.parse_string("abc")
    assert log == [("new", "abc")]

    log.clear()
    word.set_commit_action(None)
    word.parse_string("abc")
    assert log == []


def test_copy_gives_independent_commit_action_list():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "base"))
    word_copy = word.copy().add_commit_action(recorder(log, "copy"))

    word.parse_string("abc")
    assert log == [("base", "abc")]

    log.clear()
    word_copy.parse_string("abc")
    assert log == [("base", "abc"), ("copy", "abc")]


def test_commit_action_return_value_is_ignored():
    word = Word(alphas).add_commit_action(lambda toks: "REPLACED")

    result = word.parse_string("abc")

    assert result.as_list() == ["abc"]


def test_parse_actions_still_run_immediately_on_failed_parse():
    parse_action_log = []
    commit_log = []
    word = Word(alphas)
    word.add_parse_action(lambda toks: parse_action_log.append(toks[0]))
    word.add_commit_action(recorder(commit_log, "word"))
    grammar = word + Literal("!")

    # ordinary parse actions are unchanged: they still run during the parse,
    # even for a parse that ultimately fails; commit actions do not run
    with pytest.raises(pp.ParseException):
        grammar.parse_string("abc")

    assert parse_action_log == ["abc"]
    assert commit_log == []


def test_commit_action_exception_propagates_without_rollback():
    log = []

    def boom():
        raise RuntimeError("commit failed")

    grammar = Word(alphas).add_commit_action(lambda: log.append("first")) + Word(
        nums
    ).add_commit_action(boom)

    with pytest.raises(RuntimeError, match="commit failed"):
        grammar.parse_string("abc 123")

    # the commit action that already ran is not rolled back
    assert log == ["first"]


def test_packrat_cache_hit_does_not_duplicate_commit_actions():
    pp.ParserElement.enable_packrat()

    log = []
    branch1 = Literal("a").add_commit_action(recorder(log, "a1")) + Literal(
        "b"
    ).add_commit_action(recorder(log, "b"))
    branch2 = Literal("a").add_commit_action(recorder(log, "a2")) + Literal(
        "c"
    ).add_commit_action(recorder(log, "c"))
    grammar = MatchFirst([branch1, branch2])

    grammar.parse_string("ac", parse_all=True)
    assert log == [("a2", "a"), ("c", "c")]

    # a packrat cache hit on the success path must neither lose nor duplicate
    # the cached expression's commit actions
    log.clear()
    shared = Word(alphas).add_commit_action(recorder(log, "shared"))
    grammar = Or([shared + Literal(";"), shared + Literal("!")])
    grammar.parse_string("abc!", parse_all=True)
    assert log == [("shared", "abc")]


def test_packrat_lookahead_cache_hit_commits_exactly_once():
    pp.ParserElement.enable_packrat()

    log = []
    word = Literal("a").add_commit_action(recorder(log, "a"))

    # FollowedBy parses "a" at loc 0, then the And parses it again at loc 0;
    # the second parse is a packrat cache hit and must not re-commit
    (FollowedBy(word) + word).parse_string("a", parse_all=True)

    assert log == [("a", "a")]


def test_left_recursion_commit_actions():
    pp.ParserElement.enable_left_recursion()

    log = []
    num = Word(nums).add_commit_action(recorder(log, "num"))
    expr = Forward("E")
    expr <<= expr + "+" - num | num

    expr.parse_string("1+2+3", parse_all=True)
    assert log == [("num", "1"), ("num", "2"), ("num", "3")]

    # superseded recursion attempts and failed parses commit nothing
    log.clear()
    with pytest.raises(pp.ParseBaseException):
        expr.parse_string("1+2+x", parse_all=True)
    assert log == []


def test_scan_string_runs_commit_actions_per_match():
    log = []
    word = Word(alphas).add_commit_action(recorder(log, "word"))

    matches = [t[0] for t, _, _ in word.scan_string("xx abc 123 def")]

    assert matches == ["xx", "abc", "def"]
    assert log == [("word", "xx"), ("word", "abc"), ("word", "def")]


def test_nested_parse_string_gets_own_commit_scope():
    log = []
    inner = Word(alphas).add_commit_action(recorder(log, "inner"))

    def run_inner(toks):
        inner.parse_string("zzz")

    outer = Word(alphas)
    outer.add_parse_action(run_inner)
    outer.add_commit_action(recorder(log, "outer"))

    outer.parse_string("abc")

    assert log == [("inner", "zzz"), ("outer", "abc")]
