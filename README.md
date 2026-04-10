---
title: Hazardous Pigment Screener
emoji: 📚
colorFrom: yellow
colorTo: orange
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
short_description: Evidence-based screening for hazardous pigments in 19th-century cloth bindings
---

This Streamlit app is focused on screening hazardous pigment systems in 19th-century cloth-case bindings.

It now prioritises:

- nearest-reference `ΔE00` working bands for emerald-green-like, chrome-green-like, chrome-yellow-like, and mercury-red-like bindings
- optional local reference-library CSV uploads, with literature-derived chrome-yellow anchors and ISCC-NBS hue proxies used only as a fallback
- repeated LAB measurements with mean/max within-set `ΔE00` spread, with colour bands downgraded when variability crosses the active threshold
- qualitative XRF interpretation
- exact and nearest-name colour inference using a bundled ISCC-NBS LAB lookup
- safer handling guidance
- MARC-to-template export for batch review workflows

The app is intentionally conservative: it supports triage and prioritisation, not definitive pigment identification from LAB alone.

The built-in fallback references are intentionally provisional. The chrome-yellow-like class now uses historical colorimetry from Otero et al. 2012, while the emerald-green-like, chrome-green-like, and mercury-red-like classes still rely on ISCC-NBS hue proxies rather than chemically confirmed bookcloth references.

The current build also exposes class-level evidence strength from the literature, including the gap between stronger bookcloth-use evidence and weaker open colourimetry for emerald green and chrome green, plus the distinction between intrinsic toxicity and practical handling exposure.

See [METHODS.md](METHODS.md) for the current heuristic policy, traceable caveats, and source list.
