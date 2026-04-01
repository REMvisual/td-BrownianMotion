<p align="center">
  <img src="assets/banner-1280x640.png" alt="Brownian Motion" width="100%">
</p>


[![Download Latest](https://img.shields.io/github/v/release/REMvisual/td-BrownianMotion?style=for-the-badge&label=Download&color=blue)](https://github.com/REMvisual/claude-handoff/releases/latest)
![Views](https://komarev.com/ghpvc/?username=REMvisual-td-BrownianMotion&label=Views&color=brightgreen&style=for-the-badge)



Ornstein-Uhlenbeck brownian motion for TouchDesigner. A Script CHOP that generates smooth, mean-reverting procedural noise — perfect for organic camera drift, floating objects, generative motion, and anything that needs to feel alive.

## What It Does

Unlike random noise, Ornstein-Uhlenbeck motion always pulls back toward center — so you get natural, bounded wandering instead of unbounded drift. A critically-damped spring filter smooths the output for silky motion at any speed.

- **Mean-reverting** — stays within your defined range, no runaway values
- **Spring-smoothed** — implicit spring filtering removes jitter (unconditionally stable)
- **Rotation noise** — independent OU process for procedural rotation with per-axis amplitude
- **Fractal detail** — Voss-McCartney 1/f pink noise adds organic micro-motion
- **Per-axis control** — enable/disable X, Y, Z independently
- **Per-axis range** — optionally set different min/max bounds for each axis
- **Deterministic** — set a seed for repeatable motion
- **Headless-tested** — 26 automated tests validate the math at extreme parameters

## Install

**Requires TouchDesigner 2025+** (uses `CookLevel.ALWAYS` and 2025.x absTime behavior).

1. **[Download the .tox](https://github.com/REMvisual/touchdesigner-brownian-motion/releases/latest)** from Releases
2. Drag into your TouchDesigner project
3. Wire the CHOP output to whatever needs motion

Outputs six channels: `tx`, `ty`, `tz`, `rx`, `ry`, `rz`.

## Parameters

### Motion

**Range Min / Range Max** `[-1, 1]`
Output bounds. The normalized [-1, 1] motion gets mapped to this range. Default [-1, 1] means output matches the raw sim. Set to [0, 100] and your channels wander between 0 and 100.

**Speed** `1.0`
How fast time passes for the simulation. Speed=1 is real-time. Speed=5 means the OU process evolves 5x faster — more ground covered per frame. Transitions are smoothed so dragging the slider doesn't cause jumps.

**Amplitude** `1.0`
Output multiplier. 0 = silence, 1 = full range. Useful for fading motion in and out.

**Center Pull** `2.0`
How strongly the motion pulls back toward the anchor point (theta in the OU equation). High values = tight orbit around anchor, snaps back quickly. Low values = lazy wandering, barely cares about anchor. Zero = pure Brownian motion, no pull at all.

**Smoothing** `0.5`
Controls the smoothing filter on the output. 1 = very smooth, flowing motion (~2s settling). 0.5 = moderate. 0 = minimal filtering. Always applies at least a light spring — never fully raw. Changes texture, not speed.

**Detail** `0.0`
Adds high-frequency micro-motion using Voss-McCartney 1/f pink noise. 0 = off (zero cost). Higher values add more organic detail on top of the base motion.

**Detail Layers** `3`
Number of fractal octaves for the detail layer (1-5). More layers = richer high-frequency content.

### Range

**Per-Axis Range** `Off`
Toggle to set different min/max per axis. When off, all axes share the same Range Min/Max. When on, each axis gets its own bounds — useful when X needs [-180, 180] degrees but Y only needs [-10, 10].

**Range Min/Max X, Y, Z** `[-1, 1]`
Per-axis output bounds. Only visible when Per-Axis Range is on.

### Rotation

**Enable Rotation** `Off`
Master toggle for rotation noise output.

**Rotation Speed** `1.0`
Independent speed multiplier for rotation. Separate from position speed.

**Pitch / Yaw / Roll** `5 / 5 / 0`
Per-axis rotation amplitude in degrees. Set to 0 to disable that axis. Default: 5 degrees pitch and yaw, 0 roll. Shares Center Pull and Smoothing with position.

### Axes

**Affect X / Y / Z** `All on`
Per-axis enable toggles. Disabled axes output zero.

### Advanced

**Anchor X / Y / Z** `0`
The point each axis drifts toward, in normalized [-1, 1] space. Default 0 = center of range. Set Anchor X to 0.5 and `tx` will hang out in the upper half of your range.

**Seed** `0`
Random seed for reproducibility. 0 = unseeded (different every time). Any other value = deterministic — same seed always produces the same motion after Reset.

**Independent** `On`
When ON (default), each axis gets its own random noise — `tx`, `ty`, `tz` move independently in 3D. When OFF, all three axes share the same noise value each step, so they move together in lockstep (motion along the diagonal). Useful if you want correlated movement across channels.

**Reset** `Pulse`
Snap all channels back to their anchor positions and re-seed the RNG. Fresh start.

## The Math

Uses the **Ornstein-Uhlenbeck** stochastic process with the exact analytical transition (Gillespie 1996):

```
decay = exp(-theta * dt)
X = mu + (X - mu) * decay + sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta)) * Z
```

Where `theta` is center pull strength, `mu` is anchor bias, and `sigma = 0.55 * sqrt(2*theta)` is tuned so the stationary distribution fills ~80% of the [-1,1] range.

An **implicit spring** filter (Klak-style) smooths the raw OU signal. Unconditionally stable at any timestep -- no substep loops needed. The spring's natural frequency scales with speed so smoothing keeps pace with faster motion.

**Fractal detail** uses the Voss-McCartney algorithm: multiple independent OU processes updating at halved rates, summed with halving amplitude, producing true 1/f pink noise.

## License

[MIT](LICENSE) — use it however you want.
