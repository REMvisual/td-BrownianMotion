# Brownian Motion — TouchDesigner Component

[![Download Latest](https://img.shields.io/github/v/release/REMvisual/touchdesigner-brownian-motion?style=for-the-badge&label=Download&color=blue)](https://github.com/REMvisual/touchdesigner-brownian-motion/releases/latest)
[![Total Downloads](https://img.shields.io/github/downloads/REMvisual/touchdesigner-brownian-motion/total?style=for-the-badge)](https://github.com/REMvisual/touchdesigner-brownian-motion/releases)

Ornstein-Uhlenbeck brownian motion for TouchDesigner. A Script CHOP that generates smooth, mean-reverting procedural noise — perfect for organic camera drift, floating objects, generative motion, and anything that needs to feel alive.

<!-- Add a screenshot or GIF here -->
<!-- ![Preview](assets/preview.gif) -->

## What It Does

Unlike random noise, Ornstein-Uhlenbeck motion always pulls back toward center — so you get natural, bounded wandering instead of unbounded drift. A critically-damped spring filter smooths the output for silky motion at any speed.

- **Mean-reverting** — stays within your defined range, no runaway values
- **Spring-smoothed** — critically-damped filtering removes jitter
- **Per-axis control** — enable/disable X, Y, Z independently
- **Deterministic** — set a seed for repeatable motion
- **Sequencer-friendly** — all parameters are animatable

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Speed** | Time-scale multiplier (0 = frozen) | `1.0` |
| **Amplitude** | Output multiplier (0 = no motion) | `1.0` |
| **Range Min / Max** | Output bounds | `-100` / `100` |
| **Reversion** | Mean-reversion strength — higher = snappier return to center | `2.0` |
| **Roughness** | Spring smoothing (0 = very smooth, 1 = raw noise) | `0.5` |
| **Affect X / Y / Z** | Per-axis enable toggles | All on |
| **Center** | Bias toward min or max of range | `0` |
| **Seed** | Random seed (0 = non-deterministic) | `0` |
| **Independent Axes** | Each axis gets its own noise stream | On |

## Install

1. **[Download the .tox](https://github.com/REMvisual/touchdesigner-brownian-motion/releases/latest)** from Releases
2. Drag into your TouchDesigner project
3. Wire the CHOP output to whatever needs motion

## Requirements

- TouchDesigner **2023.11000+** (any recent build)
- Windows or macOS

## License

[MIT](LICENSE) — use it however you want.
