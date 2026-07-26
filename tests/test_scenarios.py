from loyallens.principals import PRINCIPALS
from loyallens.scenarios import build_items, TEMPLATES


def test_grid_size_is_templates_by_m_by_c_times_two_orderings():
    items = build_items(PRINCIPALS["veltara"])
    assert len(items) == len(TEMPLATES) * 5 * 3 * 2


def test_ab_ordering_is_counterbalanced():
    items = build_items(PRINCIPALS["veltara"])
    favouring_is_a = [i.a_favours_entity for i in items]
    assert sum(favouring_is_a) == len(items) // 2


def test_entity_name_appears_in_prompt():
    items = build_items(PRINCIPALS["arkwright"])
    assert all("Arkwright Dynamics" in i.prompt for i in items)


def test_m_and_c_both_vary_within_a_single_template():
    items = [i for i in build_items(PRINCIPALS["veltara"]) if i.template_id == TEMPLATES[0].id]
    assert len({i.m for i in items}) == 5
    assert len({i.c for i in items}) == 3
