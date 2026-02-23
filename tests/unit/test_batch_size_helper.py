from __future__ import annotations

from classification_trainer.helpers.batch_size_helper import _next_batch_size_candidate

# ---------------------------------------------------------------------------
# Phase 1: exponential doubling (last_failed is None)
# ---------------------------------------------------------------------------


def test_phase1_initial() -> None:
    assert _next_batch_size_candidate(0, None) == 1


def test_phase1_doubles() -> None:
    assert _next_batch_size_candidate(1, None) == 2
    assert _next_batch_size_candidate(2, None) == 4
    assert _next_batch_size_candidate(4, None) == 8
    assert _next_batch_size_candidate(8, None) == 16


# ---------------------------------------------------------------------------
# Phase 2: binary search (last_failed is set)
# ---------------------------------------------------------------------------


def test_phase2_midpoint() -> None:
    assert _next_batch_size_candidate(8, 16) == 12


def test_phase2_converges_upper() -> None:
    assert _next_batch_size_candidate(12, 16) == 14
    assert _next_batch_size_candidate(12, 14) == 13


def test_phase2_terminates_when_mid_equals_last_good() -> None:
    # mid = (12 + 13) // 2 == 12 == last_good → done
    assert _next_batch_size_candidate(12, 13) is None


# ---------------------------------------------------------------------------
# Full trace: true max = 12
# ---------------------------------------------------------------------------


def test_full_trace_max_12() -> None:
    sequence = []
    last_good, last_failed = 0, None
    outcomes = {1: True, 2: True, 4: True, 8: True, 16: False, 12: True, 14: False, 13: False}

    while (candidate := _next_batch_size_candidate(last_good, last_failed)) is not None:
        sequence.append(candidate)
        if outcomes[candidate]:
            last_good = candidate
        else:
            last_failed = candidate

    assert sequence == [1, 2, 4, 8, 16, 12, 14, 13]
    assert last_good == 12


# ---------------------------------------------------------------------------
# Edge case: even batch size 1 fails
# ---------------------------------------------------------------------------


def test_edge_batch_size_1_fails() -> None:
    # (0, None) → 1 (probe), fails → last_failed=1
    candidate = _next_batch_size_candidate(0, None)
    assert candidate == 1

    # (0, 1) → mid=0 == last_good → None
    assert _next_batch_size_candidate(0, 1) is None


# ---------------------------------------------------------------------------
# Edge case: exact power-of-two maximum
# ---------------------------------------------------------------------------


def test_exact_power_of_two_max() -> None:
    # True max = 8; first failure at 16
    # (8, 16) → 12, (8, 12) → 10, (8, 10) → 9, (8, 9) → None
    assert _next_batch_size_candidate(8, 16) == 12
    assert _next_batch_size_candidate(8, 12) == 10
    assert _next_batch_size_candidate(8, 10) == 9
    assert _next_batch_size_candidate(8, 9) is None
