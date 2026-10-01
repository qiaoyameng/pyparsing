#
# test_packrat_sessions.py
#
# Unit tests for parse-session scoping of packrat caches and left-recursion
# memos: thread isolation while reusing the same ParserElement, reentrant
# (nested) parsing from parse actions, cleanup on exceptional exit, manual
# reset_cache and enabling memoization while a parse is in progress.
#
import threading
import unittest

import pyparsing as pp
from pyparsing import ParseException, ParserElement


class _MemoizationTestMixin:
    def setUp(self):
        ParserElement.disable_memoization()
        self.enable_memoization()

    def tearDown(self):
        ParserElement.disable_memoization()

    def enable_memoization(self):  # pragma: no cover - overridden
        raise NotImplementedError


class TestPackratSessionScoping(_MemoizationTestMixin, unittest.TestCase):
    def enable_memoization(self):
        ParserElement.enable_packrat(force=True)

    @staticmethod
    def _backtracking_grammar():
        """
        Grammar whose shared prefix literal is attempted twice via
        backtracking (the first alternative fails on its tail). Within one
        parse the second attempt is a packrat hit, so the prefix action runs
        exactly once.
        """
        leaf = pp.Literal("a")
        calls = []

        def record_action(tokens):
            calls.append(tokens[0])
            return tokens[0]

        leaf.add_parse_action(record_action)
        grammar = (leaf + "X") | (leaf + "Y")
        return grammar, calls

    def test_single_parse_baseline(self):
        grammar, calls = self._backtracking_grammar()
        result = grammar.parse_string("aY")
        self.assertEqual(result.as_list(), ["a", "Y"])
        self.assertEqual(calls, ["a"])

    def test_sessions_do_not_reuse_cache_across_parses(self):
        grammar, calls = self._backtracking_grammar()
        grammar.parse_string("aY")
        grammar.parse_string("aX")
        grammar.parse_string("aY")
        self.assertEqual(calls, ["a", "a", "a"])

    def test_reentrant_parse_keeps_outer_cache(self):
        grammar, calls = self._backtracking_grammar()
        inner_grammar = pp.Word(pp.alphas)
        inner_calls = []

        def reentrant_action(tokens):
            inner_result = inner_grammar.parse_string("nested")
            inner_calls.append(inner_result[0])
            return tokens[0]

        grammar.exprs[0].exprs[0].add_parse_action(reentrant_action)
        result = grammar.parse_string("aY")
        self.assertEqual(result.as_list(), ["a", "Y"])
        self.assertEqual(inner_calls, ["nested"])
        # a nested top-level parse must not wipe the cache the outer parse
        # still needs when it backtracks
        self.assertEqual(calls, ["a"])

    def test_reentrant_parse_exception_does_not_clear_outer_cache(self):
        grammar, calls = self._backtracking_grammar()
        failing_inner = pp.Literal("z")

        def reentrant_failing_action(tokens):
            with self.assertRaises(ParseException):
                failing_inner.parse_string("a")
            return tokens[0]

        grammar.exprs[0].exprs[0].add_parse_action(reentrant_failing_action)
        result = grammar.parse_string("aY")
        self.assertEqual(result.as_list(), ["a", "Y"])
        self.assertEqual(calls, ["a"])

    def test_exceptional_exit_releases_session(self):
        grammar, _ = self._backtracking_grammar()
        for _ in range(5):
            with self.assertRaises(ParseException):
                grammar.parse_string("zzz")
        # failed parses must not leave state behind that changes later results
        fresh_grammar, fresh_calls = self._backtracking_grammar()
        self.assertEqual(fresh_grammar.parse_string("aY").as_list(), ["a", "Y"])
        self.assertEqual(fresh_calls, ["a"])

    def test_concurrent_same_grammar_different_inputs_with_reentrancy(self):
        # ONE shared grammar parses DIFFERENT inputs in multiple threads; each
        # action also performs one reentrant parse. Action counts and the
        # returned values are recorded per thread.
        inputs = ["aY", "aX", "aY", "aX", "aY", "aX", "aY", "aX"]
        inner_grammar = pp.Word(pp.nums)
        calls_per_thread = {}
        values_per_thread = {}
        errors = []
        barrier = threading.Barrier(len(inputs))
        lock = threading.Lock()

        leaf = pp.Literal("a")

        def shared_action(tokens):
            ident = threading.get_ident()
            with lock:
                calls_per_thread.setdefault(ident, []).append(tokens[0])
            # every action invocation performs one reentrant parse
            inner_grammar.parse_string("123")
            return tokens[0]

        leaf.add_parse_action(shared_action)
        grammar = (leaf + "X") | (leaf + "Y")

        def worker(inp):
            try:
                barrier.wait()
                result = grammar.parse_string(inp)
                values_per_thread[threading.get_ident()] = (inp, result.as_list())
            except BaseException as exc:  # pragma: no cover - failure reporting
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(inp,)) for inp in inputs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertFalse(errors, errors)
        self.assertEqual(len(values_per_thread), len(inputs))
        for ident, (inp, value) in values_per_thread.items():
            # no stale ParseResults from another thread/session
            self.assertEqual(value, ["a", inp[1]], f"foreign result in {ident}")
            # within-parse backtracking is still served by the private cache
            self.assertEqual(calls_per_thread[ident], ["a"])

    def test_reset_cache_in_other_thread_keeps_suspended_outer_cache(self):
        grammar, calls = self._backtracking_grammar()
        other_grammar, other_calls = self._backtracking_grammar()
        outer_in_action = threading.Event()
        other_done = threading.Event()

        def suspending_action(tokens):
            # tell the other thread that the outer parse has produced its
            # cached prefix, and wait while it resets and parses
            outer_in_action.set()
            other_done.wait(timeout=5)
            return tokens[0]

        grammar.exprs[0].exprs[0].add_parse_action(suspending_action)

        outcome = {}

        def other_worker():
            outer_in_action.wait(timeout=5)
            # reset_cache from a separate thread/session must not discard the
            # suspended outer parse's cached prefix result
            ParserElement.reset_cache()
            outcome["result"] = other_grammar.parse_string("aY").as_list()
            outcome["calls"] = list(other_calls)
            other_done.set()

        other = threading.Thread(target=other_worker)
        other.start()
        result = grammar.parse_string("aY")
        other.join()

        self.assertTrue(other_done.is_set())
        self.assertEqual(result.as_list(), ["a", "Y"])
        self.assertEqual(outcome["result"], ["a", "Y"])
        self.assertEqual(outcome["calls"], ["a"])
        self.assertEqual(calls, ["a"])

    def test_reset_cache_within_action_is_scoped_to_current_session(self):
        grammar, calls = self._backtracking_grammar()
        other_grammar, other_calls = self._backtracking_grammar()
        holder = {}

        def resetting_action(tokens):
            # manual reset from inside the outer parse: only this session is
            # cleared, so the outer parse re-runs the action on backtracking,
            # while the other thread's session is untouched
            ParserElement.reset_cache()
            return tokens[0]

        grammar.exprs[0].exprs[0].add_parse_action(resetting_action)

        def other_worker():
            # populate the other session, then parse again after the outer
            # parse has reset its own session; results must stay consistent
            holder["first"] = other_grammar.parse_string("aY").as_list()
            holder["first_calls"] = list(other_calls)
            holder["second"] = other_grammar.parse_string("aY").as_list()
            holder["second_calls"] = list(other_calls)

        other = threading.Thread(target=other_worker)
        other.start()
        result = grammar.parse_string("aY")
        other.join()

        self.assertEqual(result.as_list(), ["a", "Y"])
        # current session really was reset: backtracking re-ran the action ...
        self.assertEqual(calls, ["a", "a"])
        # ... but the concurrent session was unaffected
        self.assertEqual(holder["first"], ["a", "Y"])
        self.assertEqual(holder["first_calls"], ["a"])
        self.assertEqual(holder["second"], ["a", "Y"])
        self.assertEqual(holder["second_calls"], ["a", "a"])

    def test_enable_packrat_mid_parse_applies_to_current_session(self):
        ParserElement.disable_memoization()
        seen = []
        holder = {}

        def enabling_action(tokens):
            seen.append(tokens[0])
            if len(seen) == 1:
                ParserElement.enable_packrat(force=True)
                holder["nested"] = pp.Literal("q").parse_string("q")[0]
            return tokens[0]

        leaf = pp.Literal("a").add_parse_action(enabling_action)
        grammar = (leaf + "X") | (leaf + "Y")
        try:
            result = grammar.parse_string("aY")
            self.assertEqual(result.as_list(), ["a", "Y"])
            self.assertEqual(holder["nested"], "q")
        finally:
            ParserElement.disable_memoization()

    def test_scan_string_holds_one_session_per_scan(self):
        grammar, calls = self._backtracking_grammar()
        found = [tokens.as_list() for tokens, _, _ in grammar.scan_string("aY aX")]
        self.assertEqual(found, [["a", "Y"], ["a", "X"]])
        self.assertEqual(calls, ["a", "a"])

        # abandoning a generator releases its session without leaking state
        generator = grammar.scan_string("aY aX")
        next(generator)
        generator.close()
        fresh_grammar, fresh_calls = self._backtracking_grammar()
        self.assertEqual(fresh_grammar.parse_string("aY").as_list(), ["a", "Y"])
        self.assertEqual(fresh_calls, ["a"])


class TestLeftRecursionSessionScoping(_MemoizationTestMixin, unittest.TestCase):
    def enable_memoization(self):
        ParserElement.enable_left_recursion(force=True)

    @staticmethod
    def _grammar():
        expr = pp.Forward("E")
        number = pp.Word(pp.nums)
        expr <<= (expr + "+" + number) | number
        return expr

    def test_concurrent_different_inputs_no_stale_memos(self):
        # left-recursion memos are keyed by (loc, Forward, do_actions) without
        # the input string, so sharing them across parses returns stale
        # ParseResults; each thread must get its own memo table
        grammar = self._grammar()
        inputs = ["1", "1+2", "9+8+7", "3", "4+5+6+7"]
        results = {}
        errors = []
        barrier = threading.Barrier(len(inputs))
        lock = threading.Lock()

        def worker(inp):
            try:
                barrier.wait()
                results[threading.get_ident()] = (
                    inp,
                    grammar.parse_string(inp, parse_all=True).as_list(),
                )
            except BaseException as exc:  # pragma: no cover
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(inp,)) for inp in inputs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertFalse(errors, errors)
        expected = {
            "1": ["1"],
            "1+2": ["1", "+", "2"],
            "9+8+7": ["9", "+", "8", "+", "7"],
            "3": ["3"],
            "4+5+6+7": ["4", "+", "5", "+", "6", "+", "7"],
        }
        for _, (inp, value) in results.items():
            self.assertEqual(value, expected[inp], f"stale memo for {inp}")

    def test_reentrant_left_recursion_parse(self):
        grammar = self._grammar()
        inner = self._grammar()
        nested_values = []

        def reentrant_action(tokens):
            nested_values.append(
                inner.parse_string("5+6", parse_all=True).as_list()
            )
            return tokens

        # action on the base-case number of the Forward
        base_number = grammar.expr.exprs[1]
        base_number.add_parse_action(reentrant_action)

        result = grammar.parse_string("1+2", parse_all=True)
        self.assertEqual(result.as_list(), ["1", "+", "2"])
        self.assertEqual(nested_values, [["5", "+", "6"], ["5", "+", "6"]])


if __name__ == "__main__":
    unittest.main()
