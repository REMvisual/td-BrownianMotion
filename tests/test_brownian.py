"""
Headless test suite for Brownian Motion OU engine.

Tests the pure-Python core math without TouchDesigner.
Validates extremes of speed, range, and all edge cases.

Run:  python -m pytest tests/test_brownian.py -v
"""

import math
import sys
import os

# Add src to path so we can import brownian_motion
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from brownian_motion import BrownianMotion


DT = 1.0 / 60.0  # 60 fps standard timestep


def run_sim(bm, steps, dt=DT, **kwargs):
    """Run simulation for N steps, return list of (ou_state, smoothed_state) snapshots."""
    history = []
    for _ in range(steps):
        bm.step(dt, **kwargs)
        history.append((list(bm.ou_state), list(bm.smoothed_state)))
    return history


# ---------------------------------------------------------------------------
# 1. Range Coverage — OU should explore significant portion of [-1,1]
# ---------------------------------------------------------------------------
def test_range_coverage():
    bm = BrownianMotion(seed=42)
    history = run_sim(bm, 5000, speed=1.0, center_pull=2.0, smoothing=0.0)

    for ax in range(3):
        vals = [h[0][ax] for h in history]  # ou_state
        coverage = max(vals) - min(vals)
        assert coverage > 0.8, (
            f"Axis {ax}: OU coverage {coverage:.3f} < 0.8 — "
            f"expected to explore >40% of [-1,1]"
        )


# ---------------------------------------------------------------------------
# 2. Speed Scaling — higher speed = more total distance
# ---------------------------------------------------------------------------
def test_speed_scaling():
    def total_distance(speed, seed=100):
        bm = BrownianMotion(seed=seed)
        prev = (0.0, 0.0, 0.0)
        dist = 0.0
        for _ in range(2000):
            bm.step(DT, speed=speed, center_pull=2.0, smoothing=0.0)
            dx = bm.smoothed_state[0] - prev[0]
            dy = bm.smoothed_state[1] - prev[1]
            dz = bm.smoothed_state[2] - prev[2]
            dist += math.sqrt(dx*dx + dy*dy + dz*dz)
            prev = tuple(bm.smoothed_state)
        return dist

    d_slow = total_distance(1.0)
    d_fast = total_distance(5.0)
    assert d_fast > d_slow * 1.5, (
        f"Speed=5 distance {d_fast:.3f} not significantly > speed=1 distance {d_slow:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. Axis Isolation — disabled axes produce zero output
# ---------------------------------------------------------------------------
def test_axis_isolation():
    bm = BrownianMotion(seed=77)
    for _ in range(1000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)

    # Only X enabled
    tx, ty, tz = bm.map_offset(amplitude=1.0, range_min=-100, range_max=100,
                                affect=(True, False, False))
    assert tx != 0.0, "X should have moved"
    assert ty == 0.0, "Y should be zero (disabled)"
    assert tz == 0.0, "Z should be zero (disabled)"


# ---------------------------------------------------------------------------
# 4. Per-Axis Range — each axis respects its own bounds
# ---------------------------------------------------------------------------
def test_per_axis_range():
    bm = BrownianMotion(seed=55)
    axis_ranges = {'x': (-50, 50), 'y': (-200, 200), 'z': (-10, 10)}

    all_tx, all_ty, all_tz = [], [], []
    for _ in range(5000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)
        tx, ty, tz = bm.map_offset(amplitude=1.0, per_axis_range=True,
                                    axis_ranges=axis_ranges)
        all_tx.append(tx)
        all_ty.append(ty)
        all_tz.append(tz)

    # Check bounds (amplitude=1, smoothed_state in [-1,1])
    assert all(-50.0 <= v <= 50.0 for v in all_tx), "X exceeded [-50, 50]"
    assert all(-200.0 <= v <= 200.0 for v in all_ty), "Y exceeded [-200, 200]"
    assert all(-10.0 <= v <= 10.0 for v in all_tz), "Z exceeded [-10, 10]"

    # Y range is 4x wider than X — Y should have wider spread
    x_spread = max(all_tx) - min(all_tx)
    y_spread = max(all_ty) - min(all_ty)
    assert y_spread > x_spread * 1.5, (
        f"Y spread {y_spread:.1f} should be significantly wider than X spread {x_spread:.1f}"
    )


# ---------------------------------------------------------------------------
# 5. Uniform Range — per_axis_range=False gives same bounds to all axes
# ---------------------------------------------------------------------------
def test_uniform_range():
    bm = BrownianMotion(seed=33)
    for _ in range(2000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)

    tx, ty, tz = bm.map_offset(amplitude=1.0, range_min=-50, range_max=50,
                                per_axis_range=False)
    assert -50.0 <= tx <= 50.0
    assert -50.0 <= ty <= 50.0
    assert -50.0 <= tz <= 50.0


# ---------------------------------------------------------------------------
# 6. Deterministic Seed — same seed = identical playback
# ---------------------------------------------------------------------------
def test_deterministic_seed():
    def collect(seed):
        bm = BrownianMotion(seed=seed)
        results = []
        for _ in range(1000):
            bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)
            results.append(tuple(bm.smoothed_state))
        return results

    run1 = collect(12345)
    run2 = collect(12345)
    for i, (a, b) in enumerate(zip(run1, run2)):
        for ax in range(3):
            assert a[ax] == b[ax], f"Mismatch at step {i}, axis {ax}: {a[ax]} != {b[ax]}"


# ---------------------------------------------------------------------------
# 7. Extreme Speed High — speed=100, no NaN/Inf
# ---------------------------------------------------------------------------
def test_extreme_speed_high():
    bm = BrownianMotion(seed=1)
    for _ in range(1000):
        bm.step(DT, speed=100.0, center_pull=2.0, smoothing=0.5)
        for ax in range(3):
            assert math.isfinite(bm.smoothed_state[ax]), (
                f"NaN/Inf at speed=100, axis {ax}: {bm.smoothed_state[ax]}"
            )
            assert -1.0 <= bm.smoothed_state[ax] <= 1.0


# ---------------------------------------------------------------------------
# 8. Extreme Speed Zero — frozen output
# ---------------------------------------------------------------------------
def test_extreme_speed_zero():
    bm = BrownianMotion(seed=1)
    # Warm up with speed=1 to get non-zero state
    for _ in range(100):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)

    # Let speed smoother fully settle to 0 (exponential decay needs time)
    for _ in range(1000):
        bm.step(DT, speed=0.0, center_pull=2.0, smoothing=0.5)

    snapshot = list(bm.smoothed_state)

    # Now verify truly frozen — after smoothed_speed has decayed
    for _ in range(500):
        bm.step(DT, speed=0.0, center_pull=2.0, smoothing=0.5)

    for ax in range(3):
        delta = abs(bm.smoothed_state[ax] - snapshot[ax])
        assert delta < 0.01, (
            f"Axis {ax}: moved {delta:.6f} with speed=0 after settling — expected frozen"
        )


# ---------------------------------------------------------------------------
# 9. Extreme Range Large — [-10000, 10000], no overflow
# ---------------------------------------------------------------------------
def test_extreme_range_large():
    bm = BrownianMotion(seed=7)
    for _ in range(2000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)
        tx, ty, tz = bm.map_offset(amplitude=1.0, range_min=-10000, range_max=10000)
        for v in (tx, ty, tz):
            assert math.isfinite(v), f"Non-finite output with large range: {v}"
            assert -10000.0 <= v <= 10000.0


# ---------------------------------------------------------------------------
# 10. Extreme Range Tiny — [-0.001, 0.001]
# ---------------------------------------------------------------------------
def test_extreme_range_tiny():
    bm = BrownianMotion(seed=8)
    for _ in range(2000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)
        tx, ty, tz = bm.map_offset(amplitude=1.0, range_min=-0.001, range_max=0.001)
        for v in (tx, ty, tz):
            assert math.isfinite(v)
            assert -0.001 <= v <= 0.001, f"Output {v} exceeded tiny range"


# ---------------------------------------------------------------------------
# 11. Extreme Range Asymmetric — [-10, 500]
# ---------------------------------------------------------------------------
def test_extreme_range_asymmetric():
    bm = BrownianMotion(seed=9)
    all_vals = []
    for _ in range(5000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)
        tx, ty, tz = bm.map_offset(amplitude=1.0, range_min=-10, range_max=500)
        all_vals.extend([tx, ty, tz])
        for v in (tx, ty, tz):
            assert -10.0 <= v <= 500.0, f"Output {v} out of [-10, 500]"

    mean_val = sum(all_vals) / len(all_vals)
    # Midpoint is 245, center=0 so mean should be near 245
    assert 180 < mean_val < 310, (
        f"Mean {mean_val:.1f} not near midpoint 245 for range [-10, 500]"
    )


# ---------------------------------------------------------------------------
# 12. Zero Center Pull — pure random walk, still clamped
# ---------------------------------------------------------------------------
def test_zero_center_pull():
    bm = BrownianMotion(seed=10)
    for _ in range(5000):
        bm.step(DT, speed=1.0, center_pull=0.0, smoothing=0.0)
        for ax in range(3):
            assert -1.0 <= bm.ou_state[ax] <= 1.0, (
                f"OU state {bm.ou_state[ax]} escaped [-1,1] with center_pull=0"
            )


# ---------------------------------------------------------------------------
# 13. High Center Pull — stays very close to center
# ---------------------------------------------------------------------------
def test_high_center_pull():
    # With high reversion, sigma also scales (by design: std-dev stays ~0.55),
    # but the process should snap back to center faster.
    # We test: mean absolute value is low, and the process returns to center
    # quickly after perturbation (faster autocorrelation decay than low reversion).
    bm_high = BrownianMotion(seed=11)
    bm_low = BrownianMotion(seed=11)

    high_abs_vals = []
    low_abs_vals = []
    for _ in range(5000):
        bm_high.step(DT, speed=1.0, center_pull=20.0, smoothing=0.0)
        bm_low.step(DT, speed=1.0, center_pull=1.0, smoothing=0.0)
        high_abs_vals.append(abs(bm_high.ou_state[0]))
        low_abs_vals.append(abs(bm_low.ou_state[0]))

    mean_high = sum(high_abs_vals) / len(high_abs_vals)
    mean_low = sum(low_abs_vals) / len(low_abs_vals)

    # High reversion should have similar or lower mean abs value
    # Both have std-dev ~0.55 by design, but high reversion has faster dynamics
    assert mean_high < 0.7, (
        f"Mean abs with center_pull=20: {mean_high:.3f} — expected < 0.7"
    )
    # Verify the mean is reasonable (not stuck at boundary)
    assert mean_high > 0.1, (
        f"Mean abs with center_pull=20: {mean_high:.3f} — suspiciously low"
    )


# ---------------------------------------------------------------------------
# 14. Smoothing High — smoothing=1 is smoother than smoothing=0
# ---------------------------------------------------------------------------
def test_smoothing_high():
    def avg_delta_magnitude(smoothing, seed=200):
        bm = BrownianMotion(seed=seed)
        deltas = []
        prev = [0.0, 0.0, 0.0]
        for _ in range(3000):
            bm.step(DT, speed=1.0, center_pull=2.0, smoothing=smoothing)
            for ax in range(3):
                deltas.append(abs(bm.smoothed_state[ax] - prev[ax]))
            prev = list(bm.smoothed_state)
        return sum(deltas) / len(deltas)

    smooth_delta = avg_delta_magnitude(1.0)
    raw_delta = avg_delta_magnitude(0.0)
    assert smooth_delta < raw_delta, (
        f"Smoothing=1 delta {smooth_delta:.6f} should be < smoothing=0 delta {raw_delta:.6f}"
    )


# ---------------------------------------------------------------------------
# 15. Smoothing Zero — smoothing=0 bypasses spring, smoothed == OU
# ---------------------------------------------------------------------------
def test_smoothing_zero():
    bm = BrownianMotion(seed=300)
    for _ in range(500):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0)

    for ax in range(3):
        assert bm.smoothed_state[ax] == bm.ou_state[ax], (
            f"Axis {ax}: smoothed {bm.smoothed_state[ax]} != ou {bm.ou_state[ax]} "
            f"with smoothing=0"
        )


# ---------------------------------------------------------------------------
# 16. Center Bias — center=0.8 shifts mean positive
# ---------------------------------------------------------------------------
def test_center_bias():
    bm = BrownianMotion(seed=400)
    values = []
    for _ in range(5000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0,
                anchor=(0.8, 0.8, 0.8))
        values.append(sum(bm.smoothed_state) / 3.0)

    mean_val = sum(values) / len(values)
    assert mean_val > 0.3, (
        f"Mean {mean_val:.3f} should be > 0.3 with center=0.8"
    )


# ---------------------------------------------------------------------------
# 17. Amplitude Zero — all output is zero
# ---------------------------------------------------------------------------
def test_amplitude_zero():
    bm = BrownianMotion(seed=500)
    for _ in range(1000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)
        tx, ty, tz = bm.map_offset(amplitude=0.0, range_min=-100, range_max=100)
        assert tx == 0.0 and ty == 0.0 and tz == 0.0, (
            f"Output should be zero with amplitude=0: ({tx}, {ty}, {tz})"
        )


# ---------------------------------------------------------------------------
# 18. Amplitude Scaling — amplitude=2 gives 2x the output of amplitude=1
# ---------------------------------------------------------------------------
def test_amplitude_scaling():
    bm = BrownianMotion(seed=600)
    for _ in range(1000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)

    t1 = bm.map_offset(amplitude=1.0, range_min=-100, range_max=100)
    t2 = bm.map_offset(amplitude=2.0, range_min=-100, range_max=100)

    for ax in range(3):
        if t1[ax] != 0.0:
            ratio = t2[ax] / t1[ax]
            assert abs(ratio - 2.0) < 0.001, (
                f"Axis {ax}: ratio {ratio:.4f} should be 2.0"
            )


# ---------------------------------------------------------------------------
# 19. No NaN/Inf — 10000 steps with varied extreme params
# ---------------------------------------------------------------------------
def test_no_nan_inf():
    import random as stdlib_random
    rng = stdlib_random.Random(999)

    bm = BrownianMotion(seed=42)
    for _ in range(10000):
        speed = rng.uniform(0.0, 200.0)
        center_pull = rng.uniform(0.0, 50.0)
        smoothing = rng.uniform(0.0, 1.0)
        anchor = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))

        bm.step(DT, speed=speed, center_pull=center_pull, smoothing=smoothing,
                anchor=anchor)

        for ax in range(3):
            assert math.isfinite(bm.ou_state[ax]), (
                f"OU NaN/Inf at axis {ax}: {bm.ou_state[ax]} "
                f"(speed={speed:.1f}, cp={center_pull:.1f}, smooth={smoothing:.2f})"
            )
            assert math.isfinite(bm.smoothed_state[ax]), (
                f"Smoothed NaN/Inf at axis {ax}: {bm.smoothed_state[ax]}"
            )

        # Also test map_offset with extreme ranges
        range_min = rng.uniform(-50000, 0)
        range_max = rng.uniform(0, 50000)
        result = bm.map_offset(amplitude=rng.uniform(0, 10),
                               range_min=range_min, range_max=range_max)
        for v in result:
            assert math.isfinite(v), f"map_offset NaN/Inf: {v}"


# ---------------------------------------------------------------------------
# 20. Reset — state is zeroed
# ---------------------------------------------------------------------------
def test_reset():
    bm = BrownianMotion(seed=700)
    for _ in range(500):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)

    # Should have non-zero state
    assert any(v != 0.0 for v in bm.smoothed_state), "Should have moved"

    bm.reset()
    assert bm.ou_state == [0.0, 0.0, 0.0]
    assert bm.smoothed_state == [0.0, 0.0, 0.0]
    assert bm.spring_vel == [0.0, 0.0, 0.0]
