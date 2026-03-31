"""
Ornstein-Uhlenbeck Brownian Motion — Script CHOP for TouchDesigner
Outputs: tx, ty, tz
"""

import math
import random


# ═══════════════════════════════════════════════════════════════════════
# Core Engine — pure Python, no TouchDesigner dependencies
# ═══════════════════════════════════════════════════════════════════════

class BrownianMotion:
    """
    Ornstein-Uhlenbeck stochastic process with critically-damped spring filter.

    Internal state lives in [-1, 1] normalised space.
    map_offset() projects to world-space output with range/amplitude.
    """

    def __init__(self, seed=0):
        """
        Args:
            seed: Random seed. 0 = non-deterministic (time-based).
        """
        self._rng = random.Random(seed if seed != 0 else None)

        # OU state in [-1, 1] normalised space
        self.ou_state = [0.0, 0.0, 0.0]
        self.smoothed_state = [0.0, 0.0, 0.0]
        self.spring_vel = [0.0, 0.0, 0.0]

        # Speed smoother
        self.smoothed_speed = 1.0

        # Box-Muller spare
        self._has_spare = False
        self._spare = 0.0

    def reset(self):
        """Zero all state."""
        self.ou_state = [0.0, 0.0, 0.0]
        self.smoothed_state = [0.0, 0.0, 0.0]
        self.spring_vel = [0.0, 0.0, 0.0]
        self.smoothed_speed = 1.0
        self._has_spare = False
        self._spare = 0.0

    def _box_muller(self):
        """Box-Muller transform — returns one Gaussian sample N(0,1)."""
        if self._has_spare:
            self._has_spare = False
            return self._spare

        while True:
            u1 = self._rng.random()
            u2 = self._rng.random()
            if u1 > 1e-10:
                break

        mag = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self._spare = mag * math.sin(angle)
        self._has_spare = True
        return mag * math.cos(angle)

    def step(self, dt, speed=1.0, center_pull=2.0, smoothing=0.5,
             anchor=(0.0, 0.0, 0.0), independent_axes=True):
        """
        Advance the simulation by dt seconds.

        Args:
            dt: Real delta-time in seconds.
            speed: Time-scale multiplier (0 = frozen).
            center_pull: Mean-reversion strength (theta). 0-20.
            smoothing: Spring smoothing (1 = very smooth, 0 = raw noise).
            anchor: Bias per axis, each in [-1, 1].
            independent_axes: If True, each axis gets independent noise.
        """
        # Smoothing is user-facing (1=smooth, 0=raw). Internally we use roughness (0=smooth, 1=raw).
        roughness = 1.0 - smoothing

        # Smooth speed changes (~0.3s settling)
        self.smoothed_speed += (speed - self.smoothed_speed) * (1.0 - math.exp(-dt * 3.0))

        sim_dt = dt * max(self.smoothed_speed, 0.0)
        if sim_dt <= 0.0:
            return

        # ── OU substeps (speed-scaled time) ──
        # Sub-steps keep dt <= 1/120s for numerical stability
        max_step = 1.0 / 120.0
        steps = max(1, math.ceil(sim_dt / max_step))
        sub_dt = sim_dt / steps
        sqrt_sub_dt = math.sqrt(sub_dt)

        # OU sigma tuned so stationary std-dev ~ 0.55
        theta = max(center_pull, 0.0)
        sigma = 0.55 * math.sqrt(2.0 * theta) if theta > 0.0 else 0.55

        # Clamp anchor bias
        cx = max(-1.0, min(1.0, anchor[0]))
        cy = max(-1.0, min(1.0, anchor[1]))
        cz = max(-1.0, min(1.0, anchor[2]))

        for _ in range(steps):
            # Gaussian noise
            if independent_axes:
                nx = self._box_muller()
                ny = self._box_muller()
                nz = self._box_muller()
            else:
                nx = ny = nz = self._box_muller()

            # OU step: dx = theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
            self.ou_state[0] += theta * (cx - self.ou_state[0]) * sub_dt + sigma * sqrt_sub_dt * nx
            self.ou_state[1] += theta * (cy - self.ou_state[1]) * sub_dt + sigma * sqrt_sub_dt * ny
            self.ou_state[2] += theta * (cz - self.ou_state[2]) * sub_dt + sigma * sqrt_sub_dt * nz

            # Hard clamp to [-1, 1]
            self.ou_state[0] = max(-1.0, min(1.0, self.ou_state[0]))
            self.ou_state[1] = max(-1.0, min(1.0, self.ou_state[1]))
            self.ou_state[2] = max(-1.0, min(1.0, self.ou_state[2]))

        # ── Spring filter (real frame time, NOT speed-scaled) ──
        # The spring is a display-side low-pass filter. It runs at real
        # time so output stays smooth regardless of OU speed.
        # Roughness controls cutoff: 0 = ~2s settling, 1 = raw OU.
        bypass_spring = (roughness >= 1.0)
        if bypass_spring:
            self.smoothed_state = list(self.ou_state)
            self.spring_vel = [0.0, 0.0, 0.0]
        else:
            spring_speed = math.sqrt(max(self.smoothed_speed, 1.0))
            omega = 2.0 * math.exp(roughness * 3.2) * spring_speed
            omega_sq = omega * omega
            damping = 2.0 * omega

            # Substep the spring with real dt for integration stability
            spring_steps = max(1, math.ceil(dt / max_step))
            spring_dt = dt / spring_steps
            for _ in range(spring_steps):
                for ax in range(3):
                    self.spring_vel[ax] += (omega_sq * (self.ou_state[ax] - self.smoothed_state[ax])
                                            - damping * self.spring_vel[ax]) * spring_dt
                    self.smoothed_state[ax] += self.spring_vel[ax] * spring_dt

        # Safety clamp
        for ax in range(3):
            self.smoothed_state[ax] = max(-1.0, min(1.0, self.smoothed_state[ax]))

    def map_offset(self, amplitude=1.0, range_min=-100.0, range_max=100.0,
                   affect=(True, True, True), per_axis_range=False,
                   axis_ranges=None):
        """
        Map normalised [-1,1] smoothed state to world-space output.

        Args:
            amplitude: Output multiplier.
            range_min: Uniform min (used when per_axis_range=False).
            range_max: Uniform max (used when per_axis_range=False).
            affect: Tuple of 3 bools — enable per axis.
            per_axis_range: If True, use axis_ranges instead of uniform min/max.
            axis_ranges: Dict with 'x', 'y', 'z' keys, each a (min, max) tuple.
                         Required when per_axis_range=True.

        Returns:
            Tuple of (tx, ty, tz) in world-space units.
        """
        if per_axis_range and axis_ranges:
            ranges = [
                axis_ranges.get('x', (range_min, range_max)),
                axis_ranges.get('y', (range_min, range_max)),
                axis_ranges.get('z', (range_min, range_max)),
            ]
        else:
            ranges = [
                (range_min, range_max),
                (range_min, range_max),
                (range_min, range_max),
            ]

        result = [0.0, 0.0, 0.0]
        for ax in range(3):
            if not affect[ax]:
                continue
            lo, hi = ranges[ax]
            half_range = (hi - lo) * 0.5
            mid = (hi + lo) * 0.5
            result[ax] = (self.smoothed_state[ax] * half_range + mid) * amplitude

        return tuple(result)


# ═══════════════════════════════════════════════════════════════════════
# TouchDesigner Script CHOP Callbacks
# ═══════════════════════════════════════════════════════════════════════
# Everything below uses TD globals (me, absTime, project).
# Ignored when imported headlessly — TD objects won't exist.

# Module-level state — avoids scriptOp.store() cook-loop issue
_instances = {}


def onSetupParameters(scriptOp):
    """Define custom parameters on the Script CHOP."""
    page = scriptOp.appendCustomPage('Motion')
    page.appendFloat('Speed', label='Speed')
    page.appendFloat('Amplitude', label='Amplitude')
    page.appendFloat('Centerpull', label='Center Pull')
    page.appendFloat('Smoothing', label='Smoothing')

    scriptOp.par.Speed.default = 1.0
    scriptOp.par.Speed.min = 0.0
    scriptOp.par.Speed.max = 10.0
    scriptOp.par.Speed.clampMin = True
    scriptOp.par.Speed.clampMax = False

    scriptOp.par.Amplitude.default = 1.0
    scriptOp.par.Amplitude.min = 0.0
    scriptOp.par.Amplitude.max = 2.0
    scriptOp.par.Amplitude.clampMin = True
    scriptOp.par.Amplitude.clampMax = False

    scriptOp.par.Centerpull.default = 2.0
    scriptOp.par.Centerpull.min = 0.0
    scriptOp.par.Centerpull.max = 20.0
    scriptOp.par.Centerpull.clampMin = True
    scriptOp.par.Centerpull.clampMax = False

    scriptOp.par.Smoothing.default = 0.5
    scriptOp.par.Smoothing.min = 0.0
    scriptOp.par.Smoothing.max = 1.0
    scriptOp.par.Smoothing.clampMin = True
    scriptOp.par.Smoothing.clampMax = True

    # ── Range ──
    page2 = scriptOp.appendCustomPage('Range')
    page2.appendToggle('Peraxisrange', label='Per-Axis Range')
    page2.appendFloat('Rangemin', label='Range Min')
    page2.appendFloat('Rangemax', label='Range Max')
    page2.appendFloat('Rangeminx', label='Range Min X')
    page2.appendFloat('Rangemaxx', label='Range Max X')
    page2.appendFloat('Rangeminy', label='Range Min Y')
    page2.appendFloat('Rangemaxy', label='Range Max Y')
    page2.appendFloat('Rangeminz', label='Range Min Z')
    page2.appendFloat('Rangemaxz', label='Range Max Z')

    scriptOp.par.Peraxisrange.default = False
    scriptOp.par.Rangemin.default = -1.0
    scriptOp.par.Rangemax.default = 1.0
    scriptOp.par.Rangeminx.default = -1.0
    scriptOp.par.Rangemaxx.default = 1.0
    scriptOp.par.Rangeminy.default = -1.0
    scriptOp.par.Rangemaxy.default = 1.0
    scriptOp.par.Rangeminz.default = -1.0
    scriptOp.par.Rangemaxz.default = 1.0

    # Enable expressions — controls visibility based on Peraxisrange toggle
    scriptOp.par.Rangemin.enableExpr = 'not me.par.Peraxisrange'
    scriptOp.par.Rangemax.enableExpr = 'not me.par.Peraxisrange'
    scriptOp.par.Rangeminx.enableExpr = 'me.par.Peraxisrange and me.par.Affectx'
    scriptOp.par.Rangemaxx.enableExpr = 'me.par.Peraxisrange and me.par.Affectx'
    scriptOp.par.Rangeminy.enableExpr = 'me.par.Peraxisrange and me.par.Affecty'
    scriptOp.par.Rangemaxy.enableExpr = 'me.par.Peraxisrange and me.par.Affecty'
    scriptOp.par.Rangeminz.enableExpr = 'me.par.Peraxisrange and me.par.Affectz'
    scriptOp.par.Rangemaxz.enableExpr = 'me.par.Peraxisrange and me.par.Affectz'

    # ── Axes ──
    page3 = scriptOp.appendCustomPage('Axes')
    page3.appendToggle('Affectx', label='Affect X')
    page3.appendToggle('Affecty', label='Affect Y')
    page3.appendToggle('Affectz', label='Affect Z')

    scriptOp.par.Affectx.default = True
    scriptOp.par.Affecty.default = True
    scriptOp.par.Affectz.default = True

    # ── Advanced ──
    page4 = scriptOp.appendCustomPage('Advanced')
    page4.appendFloat('Anchorx', label='Anchor X')
    page4.appendFloat('Anchory', label='Anchor Y')
    page4.appendFloat('Anchorz', label='Anchor Z')
    page4.appendInt('Seed', label='Seed')
    page4.appendToggle('Independentaxes', label='Independent Axes')
    page4.appendPulse('Reset', label='Reset')

    scriptOp.par.Anchorx.default = 0.0
    scriptOp.par.Anchorx.min = -1.0
    scriptOp.par.Anchorx.max = 1.0
    scriptOp.par.Anchorx.clampMin = True
    scriptOp.par.Anchorx.clampMax = True

    scriptOp.par.Anchory.default = 0.0
    scriptOp.par.Anchory.min = -1.0
    scriptOp.par.Anchory.max = 1.0
    scriptOp.par.Anchory.clampMin = True
    scriptOp.par.Anchory.clampMax = True

    scriptOp.par.Anchorz.default = 0.0
    scriptOp.par.Anchorz.min = -1.0
    scriptOp.par.Anchorz.max = 1.0
    scriptOp.par.Anchorz.clampMin = True
    scriptOp.par.Anchorz.clampMax = True

    scriptOp.par.Seed.default = 0
    scriptOp.par.Seed.min = 0
    scriptOp.par.Seed.clampMin = True

    scriptOp.par.Independentaxes.default = True


def onGetCookLevel(scriptOp):
    """Force cook every frame — this is a continuous animation."""
    return CookLevel.ALWAYS


def onCook(scriptOp):
    """Called each cook — run OU step and output channels."""
    # Module-level state keyed by OP id — no store/fetch (avoids cook loop)
    key = scriptOp.id
    if key not in _instances:
        seed = int(scriptOp.par.Seed.eval())
        _instances[key] = BrownianMotion(seed=seed)
    bm = _instances[key]

    # Read absTime.frame to create a per-frame cook dependency
    # (store/fetch was the loop cause, not absTime itself)
    _ = absTime.frame
    dt = absTime.stepSeconds if absTime.stepSeconds > 0 else 1.0 / me.time.rate

    # Read parameters via .eval() — safe on own custom pars
    speed = scriptOp.par.Speed.eval()
    amplitude = scriptOp.par.Amplitude.eval()
    center_pull = scriptOp.par.Centerpull.eval()
    smoothing = scriptOp.par.Smoothing.eval()
    anchor = (
        scriptOp.par.Anchorx.eval(),
        scriptOp.par.Anchory.eval(),
        scriptOp.par.Anchorz.eval(),
    )
    independent = bool(scriptOp.par.Independentaxes.eval())
    per_axis = bool(scriptOp.par.Peraxisrange.eval())

    # Step the simulation
    bm.step(dt, speed=speed, center_pull=center_pull, smoothing=smoothing,
            anchor=anchor, independent_axes=independent)

    # Map to output
    if per_axis:
        axis_ranges = {
            'x': (scriptOp.par.Rangeminx.eval(), scriptOp.par.Rangemaxx.eval()),
            'y': (scriptOp.par.Rangeminy.eval(), scriptOp.par.Rangemaxy.eval()),
            'z': (scriptOp.par.Rangeminz.eval(), scriptOp.par.Rangemaxz.eval()),
        }
        tx, ty, tz = bm.map_offset(
            amplitude=amplitude,
            affect=(
                bool(scriptOp.par.Affectx.eval()),
                bool(scriptOp.par.Affecty.eval()),
                bool(scriptOp.par.Affectz.eval()),
            ),
            per_axis_range=True,
            axis_ranges=axis_ranges,
        )
    else:
        range_min = scriptOp.par.Rangemin.eval()
        range_max = scriptOp.par.Rangemax.eval()
        tx, ty, tz = bm.map_offset(
            amplitude=amplitude,
            range_min=range_min,
            range_max=range_max,
            affect=(
                bool(scriptOp.par.Affectx.eval()),
                bool(scriptOp.par.Affecty.eval()),
                bool(scriptOp.par.Affectz.eval()),
            ),
            per_axis_range=False,
        )

    # Output channels
    scriptOp.clear()
    scriptOp.appendChan('tx')[0] = tx
    scriptOp.appendChan('ty')[0] = ty
    scriptOp.appendChan('tz')[0] = tz


def onPulse(par):
    """Handle Reset pulse."""
    if par.name == 'Reset':
        key = par.owner.id
        if key in _instances:
            _instances[key].reset()
