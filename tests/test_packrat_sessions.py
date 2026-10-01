from concurrent.futures import ThreadPoolExecutor
from threading import Event
import threading

import pytest

import pyparsing as pp


@pytest.fixture
def packrat_grammar():
    pp.ParserElement.enable_packrat(cache_size_limit=None, force=True)
    yield
    pp.ParserElement.disable_memoization()


@pytest.fixture
def left_recursion_grammar():
    pp.ParserElement.enable_left_recursion(force=True)
    yield
    pp.ParserElement.disable_memoization()


def make_reentrant_grammar():
    counts = {"word": 0}
    local = threading.local()
    first_inside = Event()
    second_inside = Event()
    inside_events = [first_inside, second_inside]

    word = pp.Word(pp.nums)

    def word_action(string, location, tokens):
        counts["word"] += 1
        if string in ("1?", "2?"):
            event = inside_events[int(tokens[0]) - 1]
            inner_value = int(tokens[0]) + 2
            inner_string = f"{inner_value}x"
            local.inner_parse = True
            event.set()
            for ready_event in inside_events:
                ready_event.wait(5)
            with pytest.raises(pp.ParseException):
                grammar.parse_string(inner_string, parse_all=True)
        return [int(tokens[0]) * 10]

    word.add_parse_action(word_action)
    grammar = word + (pp.Suppress("!") | pp.Suppress("?"))
    return grammar, counts, first_inside, second_inside


def test_same_grammar_concurrent_inputs_and_one_reentry_each(packrat_grammar):
    grammar, counts, first_inside, second_inside = make_reentrant_grammar()

    def parse(value):
        return grammar.parse_string(value, parse_all=True).as_list()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outer = [pool.submit(parse, value) for value in ("1?", "2?")]
        assert first_inside.wait(5)
        assert second_inside.wait(5)
        results = [future.result(timeout=5) for future in outer]

    assert sorted(results) == [[10], [20]]
    assert counts["word"] == 4


def test_reentrant_exception_does_not_clear_enclosing_packrat_cache(packrat_grammar):
    counts = {"word": 0}
    local = threading.local()
    word = pp.Word(pp.nums)

    def word_action(tokens):
        counts["word"] += 1
        if not getattr(local, "inner_parse", False):
            local.inner_parse = True
            with pytest.raises(pp.ParseException):
                grammar.parse_string("1x", parse_all=True)
        return [int(tokens[0]) * 10]

    word.add_parse_action(word_action)
    grammar = word + (pp.Suppress("!") | pp.Suppress("?"))

    assert grammar.parse_string("1?", parse_all=True).as_list() == [10]
    assert counts["word"] == 2


def test_packrat_cache_is_not_reused_across_threads(packrat_grammar):
    word = pp.Word(pp.nums)
    producer_ready = Event()
    consumer_ready = Event()
    release_consumer = Event()
    local = threading.local()

    def start_action():
        if getattr(local, "consumer", False):
            consumer_ready.set()
            producer_ready.wait(5)
            release_consumer.wait(5)
        else:
            producer_ready.set()
            consumer_ready.wait(5)

    def word_action(tokens):
        if getattr(local, "producer", False):
            return [999]
        return [int(tokens[0])]

    start = pp.Empty().add_parse_action(start_action)
    word.add_parse_action(word_action)
    grammar = start + word

    def parse_producer():
        local.producer = True
        return grammar.parse_string("123", parse_all=True)[0]

    def parse_consumer():
        local.consumer = True
        return grammar.parse_string("123", parse_all=True)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        producer = pool.submit(parse_producer)
        consumer = pool.submit(parse_consumer)
        assert producer_ready.wait(5)
        assert consumer_ready.wait(5)
        assert producer.result(timeout=5) == 999
        release_consumer.set()
        assert consumer.result(timeout=5) == 123


def test_reset_cache_is_scoped_to_innermost_reentrant_session(packrat_grammar):
    counts = {"word": 0}
    local = threading.local()
    word = pp.Word(pp.nums)
    cached_word = pp.Word(pp.nums)

    def word_action(string, location, tokens):
        counts["word"] += 1
        if getattr(local, "inner_parse", False):
            pp.ParserElement.reset_cache()
        else:
            local.inner_parse = True
            enclosing_session = pp.ParserElement._current_parse_session()
            cached_word._parse(string, location, callPreParse=False)
            with pytest.raises(pp.ParseException):
                grammar.parse_string("1x", parse_all=True)
            assert enclosing_session is not None
            assert enclosing_session.packrat_cache is not None
            lookup = (cached_word, string, location, False, True)
            assert enclosing_session.packrat_cache.get(lookup) is not (
                enclosing_session.packrat_cache.not_in_cache
            )
        return [int(tokens[0]) * 10]

    word.add_parse_action(word_action)
    grammar = word + (pp.Suppress("!") | pp.Suppress("?"))

    assert grammar.parse_string("1?", parse_all=True).as_list() == [10]
    assert counts["word"] == 2


def test_reentrant_exception_does_not_clear_enclosing_left_recursion_memo(
    left_recursion_grammar,
):
    counts = {"number": 0}
    local = threading.local()
    expression = pp.Forward()
    number = pp.Word(pp.nums)

    def number_action(tokens):
        counts["number"] += 1
        value = int(tokens[0])
        if value == 1 and not getattr(local, "inner_parse", False):
            local.inner_parse = True
            with pytest.raises(pp.ParseException):
                expression.parse_string("1x", parse_all=True)
        return [value]

    number.add_parse_action(number_action)
    expression <<= expression + pp.Suppress("+") - number | number

    assert expression.parse_string("1", parse_all=True).as_list() == [1]
    assert counts["number"] == 2


def test_reentrant_parse_can_enable_left_recursion_without_changing_outer_session(
    packrat_grammar,
):
    counts = {"word": 0}
    local = threading.local()
    word = pp.Word(pp.nums)
    recursive_expression = pp.Forward()
    recursive_number = pp.Word(pp.nums)
    recursive_expression <<= (
        recursive_expression + pp.Suppress("+") - recursive_number
        | recursive_number
    )

    def word_action(tokens):
        counts["word"] += 1
        if not getattr(local, "inner_parse", False):
            local.inner_parse = True
            pp.ParserElement.enable_left_recursion(force=True)
            assert recursive_expression.parse_string(
                "1", parse_all=True
            ).as_list() == ["1"]
            pp.ParserElement.disable_memoization()
            assert pp.ParserElement._parse is pp.ParserElement._parseCache
        return [int(tokens[0]) * 10]

    word.add_parse_action(word_action)
    grammar = word + (pp.Suppress("!") | pp.Suppress("?"))

    assert grammar.parse_string("1?", parse_all=True).as_list() == [10]
    assert counts["word"] == 1
    assert pp.ParserElement._parse is pp.ParserElement._parseNoCache


def test_scan_string_uses_one_session_for_generator_lifetime(packrat_grammar):
    counts = {"word": 0}
    def word_action(tokens):
        counts["word"] += 1
        return int(tokens[0])

    word = pp.Word(pp.nums).add_parse_action(word_action)
    grammar = word + pp.Suppress("!")

    results = [
        (tokens[0], start, end)
        for tokens, start, end in grammar.scan_string("1! 2!")
    ]

    assert results == [(1, 0, 2), (2, 3, 5)]
    assert counts["word"] == 2
