import unittest

import pyparsing as pp
from pyparsing.core import GrammarRecursionKind


class UnknownToken(pp.Token):
    def parseImpl(self, instring, loc, do_actions=True):
        return loc, []

    def _generateDefaultName(self):
        return "unknown-token"


class UnknownWrapper(pp.TokenConverter):
    pass


class GrammarValidationTests(unittest.TestCase):
    def tearDown(self):
        pp.ParserElement.disable_memoization()

    def test_direct_no_progress_loop(self):
        grammar = pp.Forward().set_name("A")
        grammar <<= grammar

        with self.assertRaises(pp.NoProgressLoopError) as context:
            pp.validate_grammar(grammar, left_recursion=True)

        recursion = context.exception.report.recursions[0]
        self.assertIs(recursion.kind, GrammarRecursionKind.no_progress_loop)
        self.assertEqual(recursion.names, ("A", "A"))
        self.assertIn("A -> A", str(context.exception))

    def test_indirect_nullable_no_progress_loop(self):
        a = pp.Forward().set_name("A")
        b = pp.Forward().set_name("B")
        a <<= pp.Opt(b)
        b <<= pp.Opt(a)

        with self.assertRaises(pp.NoProgressLoopError) as context:
            pp.validate_grammar(a, left_recursion=True)

        recursion = context.exception.report.recursions[0]
        self.assertIs(recursion.kind, GrammarRecursionKind.no_progress_loop)
        self.assertEqual(recursion.names[0], recursion.names[-1])
        self.assertIn("A", " -> ".join(recursion.names))
        self.assertIn("B", " -> ".join(recursion.names))

    def test_left_recursive_arithmetic(self):
        expr = pp.Forward().set_name("expr")
        number = pp.Word(pp.nums).set_name("number")
        expr <<= (expr + "+" + number) | number

        with self.assertRaises(pp.LeftRecursionError) as context:
            pp.validate_grammar(expr, left_recursion=False)
        self.assertEqual(
            context.exception.report.recursions[0].kind,
            GrammarRecursionKind.left_recursion,
        )

        report = pp.validate_grammar(expr, left_recursion=True)
        self.assertTrue(report.valid)
        self.assertTrue(report.complete)
        self.assertIs(
            report.recursions[0].kind, GrammarRecursionKind.left_recursion
        )
        pp.ParserElement.enable_left_recursion(force=True)
        self.assertEqual(expr.parse_string("1+2").as_list(), ["1", "+", "2"])

    def test_terminating_right_recursion(self):
        grammar = pp.Forward().set_name("list")
        grammar <<= pp.Literal("x") + grammar | pp.Empty()

        report = pp.validate_grammar(grammar, left_recursion=False)
        self.assertTrue(report.valid)
        self.assertIs(
            report.recursions[0].kind,
            GrammarRecursionKind.terminating_recursion,
        )
        self.assertEqual(grammar.parse_string("xxx").as_list(), ["x", "x", "x"])

    def test_optional_and_zero_repetition_self_cycles_are_analyzed(self):
        for constructor in (pp.Opt, pp.ZeroOrMore):
            with self.subTest(constructor=constructor):
                grammar = pp.Forward().set_name("loop")
                grammar <<= constructor(grammar)
                with self.assertRaises(pp.NoProgressLoopError):
                    pp.validate_grammar(grammar, left_recursion=True)

    def test_nullable_ignore_cycle_is_analyzed(self):
        grammar = pp.Literal("x").set_name("ignoring")
        grammar.ignore(pp.Suppress(pp.Opt(grammar)))

        with self.assertRaises(pp.NoProgressLoopError):
            pp.validate_grammar(grammar, left_recursion=True)

    def test_unknown_custom_element_is_not_rejected(self):
        grammar = pp.Forward().set_name("custom")
        grammar <<= UnknownToken().set_name("unknown-token") + grammar

        report = pp.validate_grammar(grammar, left_recursion=False)
        self.assertTrue(report.valid)
        self.assertFalse(report.complete)
        self.assertEqual(len(report.unknown_elements), 1)
        self.assertIs(
            report.recursions[0].kind, GrammarRecursionKind.unknown_recursion
        )

    def test_unknown_custom_wrapper_self_cycle_is_unknown(self):
        grammar = pp.Forward().set_name("custom")
        grammar <<= UnknownWrapper(grammar)

        report = pp.validate_grammar(grammar, left_recursion=False)
        self.assertTrue(report.valid)
        self.assertFalse(report.complete)
        self.assertIs(
            report.recursions[0].kind, GrammarRecursionKind.unknown_recursion
        )

    def test_method_entry_point(self):
        grammar = pp.Forward().set_name("entry")
        grammar <<= grammar
        with self.assertRaises(pp.NoProgressLoopError):
            grammar.validate_grammar(left_recursion=True)


if __name__ == "__main__":
    unittest.main()
