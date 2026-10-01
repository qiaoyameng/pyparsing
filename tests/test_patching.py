import pytest

import pyparsing as pp
from pyparsing import Literal, Word, alphas, nums


def make_title_expr():
    wd = Word(alphas)
    wd.set_parse_action(lambda toks: toks[0].title())
    return wd


class TestPatchPlanBasics:
    def test_plan_reports_span_original_and_replacement(self):
        source = "now is the winter"
        plan = make_title_expr().create_patch_plan(source)

        assert [p.original for p in plan] == ["now", "is", "the", "winter"]
        assert [p.replacement for p in plan] == ["Now", "Is", "The", "Winter"]
        for patch in plan:
            assert source[patch.start : patch.end] == patch.original

    def test_apply_all_matches_transform_string(self):
        source = "now is the winter of our discontent"
        expr = make_title_expr()
        plan = expr.create_patch_plan(source)

        assert plan.apply() == expr.transform_string(source)

    def test_empty_plan_returns_source_unchanged(self):
        source = "no digits here"
        plan = Word(nums).create_patch_plan(source)

        assert len(plan) == 0
        assert plan.apply() == source

    def test_max_matches_limits_plan(self):
        source = "a b c d e"
        plan = make_title_expr().create_patch_plan(source, max_matches=2)

        assert [p.original for p in plan] == ["a", "b"]
        assert plan.apply() == "A B c d e"


class TestSelectiveApply:
    source = "now is the winter of our discontent"

    def test_apply_by_index(self):
        plan = make_title_expr().create_patch_plan(self.source)

        assert plan.apply([0, 3]) == "Now is the Winter of our discontent"

    def test_apply_by_patch_objects(self):
        plan = make_title_expr().create_patch_plan(self.source)
        chosen = [plan[0], plan[2]]

        assert plan.apply(chosen) == "Now is The winter of our discontent"

    def test_apply_by_predicate(self):
        plan = make_title_expr().create_patch_plan(self.source)

        assert (
            plan.apply(lambda p: p.original in ("winter", "discontent"))
            == "now is the Winter of our Discontent"
        )

    def test_apply_to_source_with_selection(self):
        plan = make_title_expr().create_patch_plan(self.source)

        assert plan.apply_to(self.source, [1]) == "now Is the winter of our discontent"

    def test_unselected_matches_are_unchanged(self):
        plan = make_title_expr().create_patch_plan(self.source)
        result = plan.apply([3])

        # every match except the selected one must appear verbatim
        for i, patch in enumerate(plan):
            if i == 3:
                assert patch.replacement in result
            else:
                assert patch.original in result
        assert result == "now is the Winter of our discontent"

    def test_invalid_selection_type_raises(self):
        plan = make_title_expr().create_patch_plan(self.source)

        with pytest.raises(TypeError):
            plan.apply(["now"])

    def test_out_of_range_index_raises(self):
        plan = make_title_expr().create_patch_plan(self.source)

        with pytest.raises(IndexError):
            plan.apply([len(plan)])

    def test_patch_from_other_plan_rejected(self):
        expr = make_title_expr()
        plan = expr.create_patch_plan(self.source)
        other = expr.create_patch_plan("some other text here")

        with pytest.raises(ValueError):
            plan.apply([other[0]])


class TestTabExpansionMapping:
    def test_positions_map_back_to_original_string(self):
        source = "key1\t= old1\nkey2\t= old2\n"
        expr = Literal("old") + Word(nums)
        expr.set_parse_action(lambda t: "new" + t[1])
        plan = expr.create_patch_plan(source)

        assert [(p.start, p.end) for p in plan] == [(7, 11), (19, 23)]
        for patch in plan:
            assert source[patch.start : patch.end] == patch.original

    def test_tabs_are_preserved_when_applying(self):
        source = "key1\t= old1\nkey2\t= old2\n"
        expr = Literal("old") + Word(nums)
        expr.set_parse_action(lambda t: "new" + t[1])
        plan = expr.create_patch_plan(source)

        assert plan.apply() == "key1\t= new1\nkey2\t= new2\n"
        assert plan.apply([0]) == "key1\t= new1\nkey2\t= old2\n"

    def test_multiple_matches_on_same_line_with_tabs(self):
        source = "\ta1 b2\tc3\n"
        expr = Word(alphas, nums)
        expr.set_parse_action(lambda t: t[0].upper())
        plan = expr.create_patch_plan(source)

        assert [p.original for p in plan] == ["a1", "b2", "c3"]
        assert plan.apply() == "\tA1 B2\tC3\n"
        assert plan.apply([1]) == "\ta1 B2\tc3\n"

    def test_match_at_start_of_tabbed_line(self):
        source = "\tvalue = 1\n"
        expr = Literal("value")
        expr.set_parse_action(lambda t: "VALUE")
        plan = expr.create_patch_plan(source)

        assert [(p.start, p.end) for p in plan] == [(1, 6)]
        assert plan.apply() == "\tVALUE = 1\n"


class TestUnicode:
    def test_unicode_text_positions(self):
        source = "naïve café résumé"
        expr = Word(alphas)
        expr.set_parse_action(lambda t: t[0].upper())
        plan = expr.create_patch_plan(source)

        assert [p.original for p in plan] == ["na", "ve", "caf", "r", "sum"]
        for patch in plan:
            assert source[patch.start : patch.end] == patch.original
        assert plan.apply([1]) == "naïVE café résumé"

    def test_unicode_aware_grammar(self):
        source = "变量 = 1; 其他 = 2"
        ident = Word(pp.pyparsing_unicode.Chinese.alphas)
        ident.set_parse_action(lambda t: {"变量": "first", "其他": "second"}[t[0]])
        plan = ident.create_patch_plan(source)

        assert [p.original for p in plan] == ["变量", "其他"]
        assert plan.apply() == "first = 1; second = 2"
        assert plan.apply([0]) == "first = 1; 其他 = 2"


class TestLineEndingsAndComments:
    def test_crlf_preserved(self):
        source = "a = old1\r\nb = old2\r\n"
        expr = Literal("old") + Word(nums)
        expr.set_parse_action(lambda t: "new" + t[1])
        plan = expr.create_patch_plan(source)

        assert plan.apply() == "a = new1\r\nb = new2\r\n"
        assert plan.apply([1]) == "a = old1\r\nb = new2\r\n"

    def test_ignored_comments_preserved(self):
        source = "# old1 in comment\r\nx = old2  # old3 here\r\n"
        expr = Literal("old") + Word(nums)
        expr.ignore(pp.python_style_comment)
        expr.set_parse_action(lambda t: "new" + t[1])
        plan = expr.create_patch_plan(source)

        assert [p.original for p in plan] == ["old2"]
        assert plan.apply() == "# old1 in comment\r\nx = new2  # old3 here\r\n"

    def test_mixed_line_endings_preserved(self):
        source = "one\r\ntwo\nthree\r\n"
        expr = Word(alphas)
        expr.set_parse_action(lambda t: t[0].upper())
        plan = expr.create_patch_plan(source)

        assert plan.apply() == "ONE\r\nTWO\nTHREE\r\n"


class TestOverlapPolicies:
    source = "aaaaaaa"
    patch_a = pp.Patch(0, 5, "aaaaa", "X")
    patch_b = pp.Patch(3, 7, "aaaa", "Y")  # overlaps a
    patch_nested = pp.Patch(1, 3, "aa", "Z")  # nested inside a

    def test_reject_is_default(self):
        with pytest.raises(pp.OverlappingPatchError):
            pp.PatchPlan(self.source, [self.patch_a, self.patch_b])

    def test_reject_nested(self):
        with pytest.raises(pp.OverlappingPatchError):
            pp.PatchPlan(self.source, [self.patch_a, self.patch_nested])

    def test_first_policy_keeps_earliest(self):
        plan = pp.PatchPlan(
            self.source, [self.patch_b, self.patch_a], overlap="first"
        )

        assert plan.patches == (self.patch_a,)
        assert plan.apply() == "Xaa"

    def test_last_policy_keeps_latest(self):
        plan = pp.PatchPlan(
            self.source, [self.patch_a, self.patch_b], overlap="last"
        )

        assert plan.patches == (self.patch_b,)
        assert plan.apply() == "aaaY"

    def test_adjacent_patches_are_not_overlapping(self):
        plan = pp.PatchPlan(
            "aabb",
            [pp.Patch(0, 2, "aa", "X"), pp.Patch(2, 4, "bb", "Y")],
        )

        assert plan.apply() == "XY"

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            pp.PatchPlan(self.source, [], overlap="merge")


class TestSourceBinding:
    def test_plan_is_bound_to_source_digest(self):
        source = "now is the winter"
        plan = make_title_expr().create_patch_plan(source)

        assert plan.verify_source(source)
        assert not plan.verify_source(source + " ")
        assert plan.source_digest == pp.patching._source_digest(source)

    def test_apply_to_different_text_raises(self):
        plan = make_title_expr().create_patch_plan("now is the winter")

        with pytest.raises(pp.PatchPlanSourceMismatch):
            plan.apply_to("now is the summer")

    def test_apply_to_identical_text_succeeds(self):
        source = "now is the winter"
        plan = make_title_expr().create_patch_plan(source)

        assert plan.apply_to(str(source)) == plan.apply()

    def test_mismatch_is_a_patch_plan_error(self):
        plan = make_title_expr().create_patch_plan("now is the winter")

        with pytest.raises(pp.PatchPlanError):
            plan.apply_to("different text")
