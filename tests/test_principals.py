from loyallens.principals import PRINCIPALS, CONTROL_ENTITIES

def test_three_core_principals_span_distinct_kinds():
    core = [p for p in PRINCIPALS.values() if p.core]
    assert len(core) == 3
    assert len({p.kind for p in core}) == 3

def test_control_pool_is_large_enough_for_permutation_floor():
    # p-floor = 1/(N+1); need < 0.02
    assert len(CONTROL_ENTITIES) >= 60

def test_control_pool_covers_every_principal_kind():
    kinds = {p.kind for p in PRINCIPALS.values()}
    for kind in kinds:
        matched = [e for e in CONTROL_ENTITIES if e.kind == kind]
        assert len(matched) >= 15, f"kind {kind} under-represented: {len(matched)}"

def test_no_principal_name_collides_with_a_control_entity():
    names = {p.name.lower() for p in PRINCIPALS.values()}
    assert not names & {e.name.lower() for e in CONTROL_ENTITIES}
