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

    def __init__(self, seed=0, initial_anchor=(0.0, 0.0, 0.0)):
        """
        Args:
            seed: Random seed. 0 = non-deterministic (time-based).
            initial_anchor: Starting position for OU state (eliminates cold-start drift).
        """
        self._rng = random.Random(seed if seed != 0 else None)

        # Store anchor for reset
        self._anchor = list(initial_anchor)

        # OU state in [-1, 1] normalised space — init at anchor
        self.ou_state = list(initial_anchor)
        self.smoothed_state = list(initial_anchor)
        self.spring_vel = [0.0, 0.0, 0.0]

        # Speed smoother
        self.smoothed_speed = 1.0

        # Box-Muller spare
        self._has_spare = False
        self._spare = 0.0

        # Voss-McCartney detail layers: [axis][layer] = OU state
        self._detail_states = [[0.0] * 5 for _ in range(3)]  # max 5 layers, 3 axes
        self._detail_counters = [[0] * 5 for _ in range(3)]

        # Rotation OU state (anchor is always zero for rotation)
        self.rot_ou_state = [0.0, 0.0, 0.0]
        self.rot_smoothed_state = [0.0, 0.0, 0.0]
        self.rot_spring_vel = [0.0, 0.0, 0.0]
        self.rot_smoothed_speed = 1.0
        self._rot_detail_states = [[0.0] * 5 for _ in range(3)]
        self._rot_detail_counters = [[0] * 5 for _ in range(3)]

    def reset(self, anchor=None):
        """Reset all state to anchor position."""
        if anchor is not None:
            self._anchor = list(anchor)
        self.ou_state = list(self._anchor)
        self.smoothed_state = list(self._anchor)
        self.spring_vel = [0.0, 0.0, 0.0]
        self.smoothed_speed = 1.0
        self._has_spare = False
        self._spare = 0.0
        self._detail_states = [[0.0] * 5 for _ in range(3)]
        self._detail_counters = [[0] * 5 for _ in range(3)]
        # Reset rotation
        self.rot_ou_state = [0.0, 0.0, 0.0]
        self.rot_smoothed_state = [0.0, 0.0, 0.0]
        self.rot_spring_vel = [0.0, 0.0, 0.0]
        self.rot_smoothed_speed = 1.0
        self._rot_detail_states = [[0.0] * 5 for _ in range(3)]
        self._rot_detail_counters = [[0] * 5 for _ in range(3)]

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
             anchor=(0.0, 0.0, 0.0), independent_axes=True,
             detail=0.0, detail_layers=3):
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

        # Store anchor for reset
        self._anchor = list(anchor)

        # Smooth speed changes (~0.3s settling)
        self.smoothed_speed += (speed - self.smoothed_speed) * (1.0 - math.exp(-dt * 3.0))

        sim_dt = dt * max(self.smoothed_speed, 0.0)
        if sim_dt <= 0.0:
            return

        # ── OU step (exact transition, Gillespie 1996) ──
        theta = max(center_pull, 0.0)
        sigma = 0.55 * math.sqrt(2.0 * theta) if theta > 0.0 else 0.55

        # Clamp anchor bias
        cx = max(-1.0, min(1.0, anchor[0]))
        cy = max(-1.0, min(1.0, anchor[1]))
        cz = max(-1.0, min(1.0, anchor[2]))

        if theta > 0.0:
            decay = math.exp(-theta * sim_dt)
            # Exact conditional variance: sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)
            exact_std = sigma * math.sqrt((1.0 - math.exp(-2.0 * theta * sim_dt)) / (2.0 * theta))
        else:
            decay = 1.0
            exact_std = sigma * math.sqrt(sim_dt)

        # Gaussian noise
        if independent_axes:
            nx, ny, nz = self._box_muller(), self._box_muller(), self._box_muller()
        else:
            nx = self._box_muller()
            ny, nz = nx, nx

        # Exact OU transition
        self.ou_state[0] = cx + (self.ou_state[0] - cx) * decay + exact_std * nx
        self.ou_state[1] = cy + (self.ou_state[1] - cy) * decay + exact_std * ny
        self.ou_state[2] = cz + (self.ou_state[2] - cz) * decay + exact_std * nz

        # Hard clamp to [-1, 1]
        for ax in range(3):
            self.ou_state[ax] = max(-1.0, min(1.0, self.ou_state[ax]))

        # ── Voss-McCartney 1/f detail layer ──
        # Applied BEFORE the spring so that smoothing filters detail naturally.
        # Uses sim_dt so detail freezes when speed=0.
        if detail > 0.0 and sim_dt > 0.0:
            layer_theta = 4.0
            layer_sigma = 0.55 * math.sqrt(2.0 * layer_theta)

            for ax in range(3):
                layer_sum = 0.0
                amp = 0.5
                for layer in range(min(detail_layers, 5)):
                    self._detail_counters[ax][layer] += 1
                    update_interval = 1 << layer
                    if self._detail_counters[ax][layer] >= update_interval:
                        self._detail_counters[ax][layer] = 0
                        accumulated_dt = dt * update_interval
                        if accumulated_dt > 0.0:
                            d_decay = math.exp(-layer_theta * accumulated_dt)
                            d_std = layer_sigma * math.sqrt(
                                (1.0 - math.exp(-2.0 * layer_theta * accumulated_dt))
                                / (2.0 * layer_theta))
                            self._detail_states[ax][layer] = (
                                self._detail_states[ax][layer] * d_decay
                                + d_std * self._box_muller())
                            self._detail_states[ax][layer] = max(-1.0, min(1.0,
                                self._detail_states[ax][layer]))
                    layer_sum += amp * self._detail_states[ax][layer]
                    amp *= 0.5
                # Add detail to OU state so the spring filters it
                self.ou_state[ax] = max(-1.0, min(1.0,
                    self.ou_state[ax] + layer_sum * detail * 0.15))

        # ── Spring filter (real frame time, implicit integration) ──
        # Narrowed omega range: exponent 2.0 (was 3.2), roughness capped at 0.85.
        # This keeps smoothing from drastically changing perceived speed.
        # omega range: 2 (smooth) to ~11 (rough) — 5.5x ratio vs old 21x.
        roughness = min(roughness, 0.85)

        spring_speed = math.sqrt(max(self.smoothed_speed, 1.0))
        omega = 2.0 * math.exp(roughness * 2.0) * spring_speed

        # Implicit spring (Klak-style, unconditionally stable)
        for ax in range(3):
            n1 = self.spring_vel[ax] - (self.smoothed_state[ax] - self.ou_state[ax]) * (omega * omega * dt)
            n2 = 1.0 + omega * dt
            self.spring_vel[ax] = n1 / (n2 * n2)
            self.smoothed_state[ax] += self.spring_vel[ax] * dt

        # Safety clamp
        for ax in range(3):
            self.smoothed_state[ax] = max(-1.0, min(1.0, self.smoothed_state[ax]))

    def step_rotation(self, dt, rotation_speed=1.0, center_pull=2.0, smoothing=0.5,
                      pitch=0.0, yaw=0.0, roll=0.0, detail=0.0, detail_layers=3):
        """
        Step the rotation OU process. Returns (rx, ry, rz) in degrees.

        Shares center_pull and smoothing with position but has independent speed
        and per-axis amplitude. Rotation anchor is always zero (mean-reverts to
        no-rotation).

        Args:
            dt: Real delta-time in seconds.
            rotation_speed: Time-scale multiplier for rotation.
            center_pull: Mean-reversion strength (shared with position).
            smoothing: Spring smoothing (shared with position).
            pitch: Amplitude in degrees for X rotation. 0 = disabled.
            yaw: Amplitude in degrees for Y rotation. 0 = disabled.
            roll: Amplitude in degrees for Z rotation. 0 = disabled.
            detail: Fractal detail amount (0-1).
            detail_layers: Number of Voss-McCartney layers (1-5).
        """
        roughness = 1.0 - smoothing

        # Smooth rotation speed
        self.rot_smoothed_speed += (rotation_speed - self.rot_smoothed_speed) * (
            1.0 - math.exp(-dt * 3.0))

        sim_dt = dt * max(self.rot_smoothed_speed, 0.0)
        if sim_dt <= 0.0:
            return (0.0, 0.0, 0.0)

        theta = max(center_pull, 0.0)
        sigma = 0.55 * math.sqrt(2.0 * theta) if theta > 0.0 else 0.55

        # Exact OU transition (rotation anchor is always zero)
        if theta > 0.0:
            decay = math.exp(-theta * sim_dt)
            exact_std = sigma * math.sqrt(
                (1.0 - math.exp(-2.0 * theta * sim_dt)) / (2.0 * theta))
        else:
            decay = 1.0
            exact_std = sigma * math.sqrt(sim_dt)

        nx, ny, nz = self._box_muller(), self._box_muller(), self._box_muller()

        self.rot_ou_state[0] = self.rot_ou_state[0] * decay + exact_std * nx
        self.rot_ou_state[1] = self.rot_ou_state[1] * decay + exact_std * ny
        self.rot_ou_state[2] = self.rot_ou_state[2] * decay + exact_std * nz

        for ax in range(3):
            self.rot_ou_state[ax] = max(-1.0, min(1.0, self.rot_ou_state[ax]))

        # Detail BEFORE spring (so smoothing filters it)
        if detail > 0.0 and sim_dt > 0.0:
            layer_theta = 4.0
            layer_sigma = 0.55 * math.sqrt(2.0 * layer_theta)
            for ax in range(3):
                layer_sum = 0.0
                amp = 0.5
                for layer in range(min(detail_layers, 5)):
                    self._rot_detail_counters[ax][layer] += 1
                    update_interval = 1 << layer
                    if self._rot_detail_counters[ax][layer] >= update_interval:
                        self._rot_detail_counters[ax][layer] = 0
                        accumulated_dt = dt * update_interval
                        if accumulated_dt > 0.0:
                            d_decay = math.exp(-layer_theta * accumulated_dt)
                            d_std = layer_sigma * math.sqrt(
                                (1.0 - math.exp(-2.0 * layer_theta * accumulated_dt))
                                / (2.0 * layer_theta))
                            self._rot_detail_states[ax][layer] = (
                                self._rot_detail_states[ax][layer] * d_decay
                                + d_std * self._box_muller())
                            self._rot_detail_states[ax][layer] = max(-1.0, min(1.0,
                                self._rot_detail_states[ax][layer]))
                    layer_sum += amp * self._rot_detail_states[ax][layer]
                    amp *= 0.5
                self.rot_ou_state[ax] = max(-1.0, min(1.0,
                    self.rot_ou_state[ax] + layer_sum * detail * 0.15))

        # Spring (narrowed omega range, roughness capped at 0.85)
        roughness = min(roughness, 0.85)
        spring_speed = math.sqrt(max(self.rot_smoothed_speed, 1.0))
        omega = 2.0 * math.exp(roughness * 2.0) * spring_speed
        for ax in range(3):
            n1 = self.rot_spring_vel[ax] - (
                self.rot_smoothed_state[ax] - self.rot_ou_state[ax]) * (omega * omega * dt)
            n2 = 1.0 + omega * dt
            self.rot_spring_vel[ax] = n1 / (n2 * n2)
            self.rot_smoothed_state[ax] += self.rot_spring_vel[ax] * dt

        for ax in range(3):
            self.rot_smoothed_state[ax] = max(-1.0, min(1.0, self.rot_smoothed_state[ax]))

        # Map to degrees — amplitude per axis (0 = disabled)
        amplitudes = [pitch, yaw, roll]
        return tuple(self.rot_smoothed_state[ax] * amplitudes[ax] for ax in range(3))

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

    page.appendFloat('Detail', label='Detail')
    page.appendInt('Detaillayers', label='Detail Layers')

    scriptOp.par.Detail.default = 0.0
    scriptOp.par.Detail.min = 0.0
    scriptOp.par.Detail.max = 1.0
    scriptOp.par.Detail.clampMin = True
    scriptOp.par.Detail.clampMax = True

    scriptOp.par.Detaillayers.default = 3
    scriptOp.par.Detaillayers.min = 1
    scriptOp.par.Detaillayers.max = 5
    scriptOp.par.Detaillayers.clampMin = True
    scriptOp.par.Detaillayers.clampMax = True

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

    # ── Rotation ──
    page_rot = scriptOp.appendCustomPage('Rotation')
    page_rot.appendToggle('Enablerotation', label='Enable Rotation')
    page_rot.appendFloat('Rotationspeed', label='Rotation Speed')
    page_rot.appendFloat('Rotationpitch', label='Pitch')
    page_rot.appendFloat('Rotationyaw', label='Yaw')
    page_rot.appendFloat('Rotationroll', label='Roll')

    scriptOp.par.Enablerotation.default = False

    scriptOp.par.Rotationspeed.default = 1.0
    scriptOp.par.Rotationspeed.min = 0.0
    scriptOp.par.Rotationspeed.max = 10.0
    scriptOp.par.Rotationspeed.clampMin = True
    scriptOp.par.Rotationspeed.clampMax = False

    scriptOp.par.Rotationpitch.default = 5.0
    scriptOp.par.Rotationpitch.min = -180.0
    scriptOp.par.Rotationpitch.max = 180.0
    scriptOp.par.Rotationpitch.clampMin = False
    scriptOp.par.Rotationpitch.clampMax = False

    scriptOp.par.Rotationyaw.default = 5.0
    scriptOp.par.Rotationyaw.min = -180.0
    scriptOp.par.Rotationyaw.max = 180.0
    scriptOp.par.Rotationyaw.clampMin = False
    scriptOp.par.Rotationyaw.clampMax = False

    scriptOp.par.Rotationroll.default = 0.0
    scriptOp.par.Rotationroll.min = -180.0
    scriptOp.par.Rotationroll.max = 180.0
    scriptOp.par.Rotationroll.clampMin = False
    scriptOp.par.Rotationroll.clampMax = False

    # Enable rotation params only when toggle is on
    scriptOp.par.Rotationspeed.enableExpr = 'me.par.Enablerotation'
    scriptOp.par.Rotationpitch.enableExpr = 'me.par.Enablerotation'
    scriptOp.par.Rotationyaw.enableExpr = 'me.par.Enablerotation'
    scriptOp.par.Rotationroll.enableExpr = 'me.par.Enablerotation'

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
    detail = scriptOp.par.Detail.eval()
    detail_layers = int(scriptOp.par.Detaillayers.eval())

    # Step the simulation
    bm.step(dt, speed=speed, center_pull=center_pull, smoothing=smoothing,
            anchor=anchor, independent_axes=independent,
            detail=detail, detail_layers=detail_layers)

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

    # Always output 6 channels (zeros when rotation disabled) for stable downstream refs
    scriptOp.clear()
    scriptOp.appendChan('tx')[0] = tx
    scriptOp.appendChan('ty')[0] = ty
    scriptOp.appendChan('tz')[0] = tz

    rotation_enabled = bool(scriptOp.par.Enablerotation.eval())
    if rotation_enabled:
        rot_speed = scriptOp.par.Rotationspeed.eval()
        pitch = scriptOp.par.Rotationpitch.eval()
        yaw = scriptOp.par.Rotationyaw.eval()
        roll = scriptOp.par.Rotationroll.eval()
        rx, ry, rz = bm.step_rotation(dt, rotation_speed=rot_speed,
                                        center_pull=center_pull, smoothing=smoothing,
                                        pitch=pitch, yaw=yaw, roll=roll,
                                        detail=detail, detail_layers=detail_layers)
    else:
        rx, ry, rz = 0.0, 0.0, 0.0

    scriptOp.appendChan('rx')[0] = rx
    scriptOp.appendChan('ry')[0] = ry
    scriptOp.appendChan('rz')[0] = rz


def onPulse(par):
    """Handle Reset pulse."""
    if par.name == 'Reset':
        key = par.owner.id
        if key in _instances:
            _instances[key].reset()
