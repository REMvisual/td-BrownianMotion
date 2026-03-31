"""
Deep math validation for Brownian Motion OU engine.
Verifies statistical properties, not just bounds.
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from brownian_motion import BrownianMotion

DT = 1.0 / 60.0
N = 50000  # long runs for statistical accuracy


def stats(values):
    """Mean, std, min, max of a list."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var), min(values), max(values)


def autocorrelation(values, lag):
    """Compute autocorrelation at given lag."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var == 0:
        return 0
    cov = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return cov / var


def smoothness_metric(values):
    """Average absolute frame-to-frame delta — lower = smoother."""
    deltas = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    return sum(deltas) / len(deltas)


print("=" * 70)
print("BROWNIAN MOTION — DEEP MATH VALIDATION")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. OU STATIONARY DISTRIBUTION
# Theory: std_dev = sigma / sqrt(2*theta) = 0.55 (by design)
# But hard clamp at [-1,1] truncates the tails
# ─────────────────────────────────────────────────────────────────────
print("\n1. OU STATIONARY DISTRIBUTION (smoothing=0 to bypass spring)")
for theta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    bm = BrownianMotion(seed=42)
    vals = []
    for _ in range(N):
        bm.step(DT, speed=1.0, center_pull=theta, smoothing=0.0)
        vals.append(bm.ou_state[0])
    mean, std, lo, hi = stats(vals)
    sigma = 0.55 * math.sqrt(2 * theta)
    theoretical_std = sigma / math.sqrt(2 * theta)  # = 0.55 always
    print(f"  theta={theta:5.1f}  sigma={sigma:.3f}  measured_std={std:.3f}  "
          f"theoretical={theoretical_std:.3f}  mean={mean:+.3f}  range=[{lo:.3f}, {hi:.3f}]")

# ─────────────────────────────────────────────────────────────────────
# 2. REVERSION SPEED — autocorrelation decay
# Theory: OU autocorr at lag t = exp(-theta * t)
# Higher theta = faster decorrelation
# ─────────────────────────────────────────────────────────────────────
print("\n2. REVERSION SPEED (autocorrelation at lag=30 frames = 0.5s)")
for theta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    bm = BrownianMotion(seed=42)
    vals = []
    for _ in range(N):
        bm.step(DT, speed=1.0, center_pull=theta, smoothing=0.0)
        vals.append(bm.ou_state[0])
    lag_frames = 30
    lag_time = lag_frames * DT
    ac = autocorrelation(vals, lag_frames)
    theoretical_ac = math.exp(-theta * lag_time)
    print(f"  theta={theta:5.1f}  autocorr={ac:.3f}  theoretical={theoretical_ac:.3f}  "
          f"ratio={ac/theoretical_ac:.2f}" if theoretical_ac > 0.001 else
          f"  theta={theta:5.1f}  autocorr={ac:.3f}  theoretical~0")

# ─────────────────────────────────────────────────────────────────────
# 3. CENTER BIAS
# With center=c, the OU mean should converge to c
# ─────────────────────────────────────────────────────────────────────
print("\n3. CENTER BIAS (center_pull=2, smoothing=0)")
for center in [-0.8, -0.4, 0.0, 0.4, 0.8]:
    bm = BrownianMotion(seed=42)
    vals = []
    for _ in range(N):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.0, anchor=(center, 0, 0))
        vals.append(bm.ou_state[0])
    mean, std, _, _ = stats(vals)
    error = abs(mean - center)
    print(f"  center={center:+.1f}  measured_mean={mean:+.3f}  error={error:.3f}  std={std:.3f}")

# ─────────────────────────────────────────────────────────────────────
# 4. SPEED MULTIPLIER — total path distance should scale ~linearly
# ─────────────────────────────────────────────────────────────────────
print("\n4. SPEED MULTIPLIER (smoothing=0, 10000 steps)")
base_dist = None
for speed in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    bm = BrownianMotion(seed=42)
    total_dist = 0
    prev = 0
    for _ in range(10000):
        bm.step(DT, speed=speed, center_pull=2.0, smoothing=0.0)
        total_dist += abs(bm.ou_state[0] - prev)
        prev = bm.ou_state[0]
    if base_dist is None:
        base_dist = total_dist
    ratio = total_dist / base_dist
    print(f"  speed={speed:5.2f}  total_dist={total_dist:.1f}  ratio_vs_0.25={ratio:.2f}x")

# ─────────────────────────────────────────────────────────────────────
# 5. SPRING SMOOTHNESS — compare deltas at different roughness
# ─────────────────────────────────────────────────────────────────────
print("\n5. SPRING SMOOTHNESS (speed=1, center_pull=2)")
for smoothing in [1.0, 0.75, 0.5, 0.25, 0.0]:
    bm = BrownianMotion(seed=42)
    vals = []
    for _ in range(10000):
        bm.step(DT, speed=1.0, center_pull=2.0, smoothing=smoothing)
        vals.append(bm.smoothed_state[0])
    sm = smoothness_metric(vals)
    mean, std, lo, hi = stats(vals)
    print(f"  smoothing={smoothing:.2f}  avg_delta={sm:.5f}  std={std:.3f}  range=[{lo:.3f}, {hi:.3f}]")

# ─────────────────────────────────────────────────────────────────────
# 6. SPRING AT HIGH SPEED — verify decoupled spring stays smooth
# ─────────────────────────────────────────────────────────────────────
print("\n6. SPRING AT HIGH SPEED (smoothing=0.7)")
for speed in [1.0, 5.0, 10.0, 50.0, 100.0]:
    bm = BrownianMotion(seed=42)
    vals = []
    for _ in range(5000):
        bm.step(DT, speed=speed, center_pull=2.0, smoothing=0.7)
        vals.append(bm.smoothed_state[0])
    sm = smoothness_metric(vals)
    mean, std, lo, hi = stats(vals)
    omega = 2.0 * math.exp(0.3 * 3.2) * math.sqrt(max(speed, 1.0))
    print(f"  speed={speed:5.1f}  avg_delta={sm:.5f}  std={std:.3f}  omega={omega:.1f}")

# ─────────────────────────────────────────────────────────────────────
# 7. SPRING SETTLING TIME
# Step response: set ou_state to 1.0 instantly, measure how long
# smoothed_state takes to reach 0.95
# ─────────────────────────────────────────────────────────────────────
print("\n7. SPRING SETTLING TIME (step response to 0.95)")
for smoothing in [1.0, 0.75, 0.5, 0.25, 0.1]:
    roughness = 1.0 - smoothing
    bm = BrownianMotion(seed=1)
    # Force OU to 1.0 and hold it there
    bm.ou_state = [1.0, 0.0, 0.0]
    bm.smoothed_state = [0.0, 0.0, 0.0]
    bm.spring_vel = [0.0, 0.0, 0.0]
    settle_frame = -1
    for i in range(600):  # 10 seconds
        bm.ou_state = [1.0, 0.0, 0.0]  # hold target
        # Only run spring, not OU
        bypass = roughness >= 1.0
        if not bypass:
            omega = 2.0 * math.exp(roughness * 3.2)
            omega_sq = omega * omega
            damping = 2.0 * omega
            bm.spring_vel[0] += (omega_sq * (1.0 - bm.smoothed_state[0]) - damping * bm.spring_vel[0]) * DT
            bm.smoothed_state[0] += bm.spring_vel[0] * DT
        if bm.smoothed_state[0] >= 0.95 and settle_frame < 0:
            settle_frame = i
    settle_time = settle_frame * DT if settle_frame >= 0 else float('inf')
    print(f"  smoothing={smoothing:.2f}  settle_95%={settle_time:.3f}s  ({settle_frame} frames)  "
          f"omega={2.0 * math.exp(roughness * 3.2):.1f}")

# ─────────────────────────────────────────────────────────────────────
# 8. AMPLITUDE LINEARITY
# ─────────────────────────────────────────────────────────────────────
print("\n8. AMPLITUDE LINEARITY")
bm = BrownianMotion(seed=42)
for _ in range(5000):
    bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)
for amp in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]:
    tx, _, _ = bm.map_offset(amplitude=amp, range_min=-1, range_max=1)
    expected = bm.smoothed_state[0] * amp
    error = abs(tx - expected)
    print(f"  amp={amp:.2f}  output={tx:+.5f}  expected={expected:+.5f}  error={error:.1e}")

# ─────────────────────────────────────────────────────────────────────
# 9. RANGE MAPPING CORRECTNESS
# ─────────────────────────────────────────────────────────────────────
print("\n9. RANGE MAPPING")
bm = BrownianMotion(seed=42)
for _ in range(5000):
    bm.step(DT, speed=1.0, center_pull=2.0, smoothing=0.5)
state = bm.smoothed_state[0]
print(f"  smoothed_state[0] = {state:.5f}")
for rmin, rmax in [(-1, 1), (-100, 100), (0, 10), (-50, 200)]:
    tx, _, _ = bm.map_offset(amplitude=1.0, range_min=rmin, range_max=rmax)
    half = (rmax - rmin) / 2
    mid = (rmax + rmin) / 2
    expected = state * half + mid
    error = abs(tx - expected)
    print(f"  range=[{rmin:+.0f},{rmax:+.0f}]  output={tx:+.3f}  expected={expected:+.3f}  error={error:.1e}")

# ─────────────────────────────────────────────────────────────────────
# 10. BOX-MULLER GAUSSIAN QUALITY
# ─────────────────────────────────────────────────────────────────────
print("\n10. BOX-MULLER GAUSSIAN QUALITY (100000 samples)")
bm = BrownianMotion(seed=42)
samples = [bm._box_muller() for _ in range(100000)]
mean, std, lo, hi = stats(samples)
# Check normality: ~68% within 1 std, ~95% within 2 std
within_1 = sum(1 for s in samples if abs(s) < 1) / len(samples)
within_2 = sum(1 for s in samples if abs(s) < 2) / len(samples)
within_3 = sum(1 for s in samples if abs(s) < 3) / len(samples)
print(f"  mean={mean:+.4f} (expect 0)  std={std:.4f} (expect 1)")
print(f"  within 1sd: {within_1:.3f} (expect 0.683)")
print(f"  within 2sd: {within_2:.3f} (expect 0.954)")
print(f"  within 3sd: {within_3:.3f} (expect 0.997)")

# ─────────────────────────────────────────────────────────────────────
# 11. SUBSTEP COUNT VERIFICATION
# ─────────────────────────────────────────────────────────────────────
print("\n11. SUBSTEP COUNTS")
for speed in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    sim_dt = DT * speed
    max_step = 1.0 / 120.0
    steps = max(1, math.ceil(sim_dt / max_step))
    sub_dt = sim_dt / steps
    print(f"  speed={speed:5.1f}  sim_dt={sim_dt:.5f}  steps={steps:3d}  sub_dt={sub_dt:.5f}  "
          f"(max={max_step:.5f})")

# ─────────────────────────────────────────────────────────────────────
# 12. UE5 COMPARISON — sigma formula
# ─────────────────────────────────────────────────────────────────────
print("\n12. UE5 SIGMA COMPARISON")
print("  UE5: sigma = 0.55 * sqrt(2 * theta)")
print("  TD:  sigma = 0.55 * sqrt(2 * theta)")
for theta in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    sigma = 0.55 * math.sqrt(2.0 * theta)
    theoretical_std = sigma / math.sqrt(2.0 * theta)
    print(f"  theta={theta:5.1f}  sigma={sigma:.4f}  stationary_std={theoretical_std:.4f}  "
          f"2sigma_coverage={2*theoretical_std:.2f} of [-1,1]")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
