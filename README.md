# Synesthesia (Web) — Orb + Harmonic Wave + In-Orb FFT

- Bigger central orb, with **white FFT** rendered from the horizontal center axis **upward** plus **slanted tails** below (variable translucency).
- **Yellow harmonic wave** traverses **only across the orb** (left ↔ right), amplitude driven by RMS + band energies; higher-impact overlay for drama.
- Outer radial halo retained. Press **H** to toggle the UI.

## Run
```bash
npm install
npm run dev
# or build:
npm run build
```


## ✨ What's New (Visualizer Pro)
- **Swarm of Boids** that flock to a moving target derived from the **predicted next note**.
- **Beat‑synced sparkles** using **spectral flux** for onsets.
- **Dual pitch engine** (autocorrelation + simplified cepstrum “second‑order FFT”) fused by confidence.
- **Orb + Rays mode** that turns the spectrum into a radial field.
- Live HUD showing **current pitch** and **predicted next** note.

### Dev
```bash
npm install
npm run dev
# Toggle Pro/Classic via the checkbox in the UI.
```
