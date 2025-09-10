import { useEffect, useRef } from "react";
import type { MicAnalyser } from "../audio/useMicAnalyser";

type Props = {
  analyser: MicAnalyser;
  className?: string;
  smoothing?: number;
  sensitivity?: number;
  /** 0..1 normalized; controls leftward sinusoid speed */
  speed?: number;
};

function dbToUnit(db: number): number {
  const min = -100,
    max = -10;
  const clamped = Math.max(min, Math.min(max, db));
  return (clamped - min) / (max - min);
}
const DEG = Math.PI / 180;
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

// deterministic, seedable PRNG (per-frame) for stable bar alpha
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function Visualizer({
  analyser,
  className,
  smoothing = 0.85,
  sensitivity = 1.0,
  speed = 0.5,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const emaSpectrumRef = useRef<Float32Array | null>(null);
  const timeBufRef = useRef<Uint8Array | null>(null);
  const freqBufRef = useRef<Float32Array | null>(null);
  const tRef = useRef<number>(0);
  const fpsRef = useRef<number>(60);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d", { alpha: false })!;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    function resize() {
      const { clientWidth, clientHeight } = canvas;
      canvas.width = Math.floor(clientWidth * dpr);
      canvas.height = Math.floor(clientHeight * dpr);
    }
    const onResize = () => resize();
    const onVis = () => {
      if (document.visibilityState === "hidden") cancel();
      else loop();
    };
    resize();
    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVis);

    let last = performance.now();
    const loop = () => {
      const node = analyser.node();
      const W = canvas.width,
        H = canvas.height;
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      tRef.current += dt;

      // FPS (EMA)
      const fpsInstant = dt > 0 ? 1 / dt : fpsRef.current;
      fpsRef.current = fpsRef.current * 0.85 + fpsInstant * 0.15;

      if (!node) {
        ctx.fillStyle = "#0a0e2a";
        ctx.fillRect(0, 0, W, H);
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const bins = analyser.frequencyBinCount();
      if (!freqBufRef.current || freqBufRef.current.length !== bins) {
        freqBufRef.current = new Float32Array(bins);
        emaSpectrumRef.current = new Float32Array(bins);
      }
      if (!timeBufRef.current || timeBufRef.current.length !== node.fftSize) {
        timeBufRef.current = new Uint8Array(node.fftSize);
      }

      analyser.getFloatFrequency(freqBufRef.current!);
      analyser.getTimeDomain(timeBufRef.current!);

      const freq = freqBufRef.current!;
      const ema = emaSpectrumRef.current!;
      for (let i = 0; i < freq.length; i++) {
        const u = dbToUnit(freq[i]) * sensitivity;
        ema[i] = ema[i] * smoothing + u * (1 - smoothing);
      }

      draw(
        ctx,
        W,
        H,
        ema,
        timeBufRef.current!,
        node.fftSize,
        tRef.current,
        analyser.sampleRate() ?? 48000,
        fpsRef.current,
        speed
      );
      rafRef.current = requestAnimationFrame(loop);
    };

    const cancel = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
    loop();
    return () => {
      cancel();
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [analyser, smoothing, sensitivity, speed]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}

function draw(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  spectrum: Float32Array,
  time: Uint8Array,
  fftSize: number,
  t: number,
  sampleRate: number,
  fps: number,
  speedNorm: number
) {
  ctx.fillStyle = "#0a0e2a";
  ctx.fillRect(0, 0, W, H);
  const cx = W * 0.5,
    cy = H * 0.5;
  const orbR = Math.min(W, H) * 0.35;

  // RMS loudness
  let rms = 0;
  for (let i = 0; i < time.length; i++) {
    const x = (time[i] - 128) / 128;
    rms += x * x;
  }
  rms = Math.sqrt(rms / time.length);

  // Outer halo
  const baseR_out = orbR * 1.1,
    maxR = Math.min(W, H) * 0.52;
  const n = spectrum.length,
    step = Math.max(1, Math.floor(n / 240));
  ctx.save();
  ctx.translate(cx, cy);
  for (let i = 0; i < n; i += step) {
    const pct = i / n,
      mag = Math.pow(spectrum[i], 1.35);
    const r = baseR_out + mag * (maxR - baseR_out);
    const th = pct * Math.PI * 2;
    const x1 = Math.cos(th) * baseR_out,
      y1 = Math.sin(th) * baseR_out;
    const x2 = Math.cos(th) * r,
      y2 = Math.sin(th) * r;
    const hue = 210 + pct * 120;
    ctx.strokeStyle = `hsl(${hue},70%,${Math.min(78, 35 + mag * 55)}%)`;
    ctx.lineWidth = Math.max(1, mag * 5.0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
  ctx.restore();

  // Orb
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, orbR * 1.35);
  glow.addColorStop(0, "rgba(255,216,77,0.22)");
  glow.addColorStop(1, "rgba(255,216,77,0.00)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, orbR * 1.35, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(5,8,26,0.92)";
  ctx.beginPath();
  ctx.arc(cx, cy, orbR, 0, Math.PI * 2);
  ctx.fill();

  // In-orb FFT (white upward + slanted bluish downward)
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, orbR * 0.97, 0, Math.PI * 2);
  ctx.clip();

  const bars = Math.min(160, spectrum.length);
  const stride = Math.max(1, Math.floor(spectrum.length / bars));
  const halfW = orbR * 0.92;
  const left = cx - halfW;
  const width = 2 * halfW;
  const barW = width / bars;
  const axisY = cy;

  const SCALE = 1.25;
  const tilt = -18 * DEG;

  // seed once per frame so each bar's alpha is stable within the frame
  const seed =
    ((Math.floor(t * (0.1 + 1.9 * Math.max(0, Math.min(1, speedNorm))) * 60) ^
      0x9e3779b9) >>>
      0) >>> 0;
  const rngFor = (i: number) => mulberry32(seed ^ (i + 17));

  ctx.globalCompositeOperation = "lighter";
  for (let bi = 0, i = 0; bi < bars; bi++, i += stride) {
    const u = clamp01(spectrum[i] * SCALE);
    const rnd = rngFor(bi)(); // deterministic [0,1)
    const alpha = 0.18 + 0.55 * u + 0.25 * rnd;

    const x = left + bi * barW;
    const w = Math.max(2, barW * 0.8);

    const hUp = u * orbR * 0.9;
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.fillRect(x, axisY - hUp, w, hUp);

    const hDown = u * orbR * 0.25;
    const anchorX = x + w;

    ctx.save();
    ctx.translate(anchorX, axisY);
    ctx.rotate(tilt);
    ctx.fillStyle = `rgba(170,188,255,${alpha * 0.7})`;
    ctx.fillRect(
      -Math.max(2, barW * 0.75) / 2,
      0,
      Math.max(2, barW * 0.75),
      hDown
    );
    ctx.restore();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.restore();

  // Horizontal axis across orb
  ctx.strokeStyle = "rgba(255,255,255,0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - orbR, cy);
  ctx.lineTo(cx + orbR, cy);
  ctx.stroke();

  // --- LITERAL TRANSLATION OF YOUR SWIFT SINUSOID (constrained to orb width) ---

  // Time / motion
  const runSec = t;
  const baseSpeed = 0.1 + 1.9 * Math.max(0, Math.min(1, speedNorm)); // 0..1 → ~0.1..2.0

  // FFT stats used by the label & color blend
  const hzNy = sampleRate / 2;
  const binHz = hzNy / Math.max(1, spectrum.length);
  let fft_max = 1e-6;
  let peak_idx = 0;
  for (let i = 0; i < spectrum.length; i++) {
    if (spectrum[i] > fft_max) {
      fft_max = spectrum[i];
      peak_idx = i;
    }
  }
  const peak_frequency = peak_idx * binHz;
  const amplitude = rms;

  // --- Yellow sinusoid only inside orb (bouncier when loud) ---

// Keep your band helpers
function band(h0:number,h1:number){
  const i0 = Math.max(0, Math.floor(h0/binHz));
  const i1 = Math.min(spectrum.length-1, Math.ceil(h1/binHz));
  let s = 0;
  for (let i=i0;i<=i1;i++) s += spectrum[i];
  return s / Math.max(1, i1-i0+1);
}

const A1 = band(80,300);
const A2 = band(300,1200);
const A3 = band(1200,3000);
const A4 = band(3000,6000);

// Loudness / HF energy drive the “bounce”
const hiEnergy = 0.5*A3 + 0.8*A4;
const ampBoost = Math.max(0, Math.min(1, 1.6*rms + 1.2*hiEnergy)); // 0..1

// Amplitude & motion
const amp    = (0.30 + 1.05*rms) * (1.0 + 0.55*ampBoost) * orbR * 0.95;
const omega  = 2*Math.PI * (0.45 * (1.0 + 0.9*ampBoost));           // time freq
const kx     = 2*Math.PI * (0.55 + 0.9*A2) * (1.0 + 0.8*ampBoost);  // spatial freq
const ksp    = kx / (2*orbR);                                       // per pixel

ctx.save();
ctx.beginPath();
ctx.arc(cx, cy, orbR*0.98, 0, Math.PI*2);
ctx.clip();
ctx.lineJoin = 'round';
ctx.lineCap  = 'round';

const R = cx - orbR;
const steps = Math.floor(2*orbR);

// MAIN bright stroke — bias to base sine; harmonics scale softly with loudness
ctx.strokeStyle = 'rgba(255,216,77,0.95)';
ctx.lineWidth   = Math.max(2.2, 2.6 + 2.0*(A1 + A2) + 2.2*ampBoost);

ctx.beginPath();
for (let s=0; s<=steps; s++){
  const x = R + (s/steps)*(2*orbR);

  // RIGHT -> LEFT motion: phase = ω t - k x
  const ph = omega*t - ksp*(x - R);

  // More “reciprocal/sinusoidy”: dominant base sine + gentle partials
  const partialMix = 0.55*ampBoost; // how much color to add when loud
  let y0 = (
      1.00 * Math.sin(ph) +
      partialMix * (0.40*Math.sin(2*ph + 0.20) +
                    0.25*Math.sin(3*ph + 0.85) +
                    0.15*Math.sin(4*ph + 1.40))
  );

  // Tiny HF shimmer only when very loud; keep symmetric (no DC)
  if (ampBoost > 0.6) {
    const e = (ampBoost-0.6)/0.4; // 0..1
    y0 += 0.12*e*Math.sin(5*ph + 0.9) + 0.08*e*Math.sin(6*ph + 1.7);
  }

  const y = cy + y0*amp;
  if (s===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
}
ctx.stroke();

// ECHO TRAILS — spaced in time; still move right->left; decay slower when loud
const echoCount = 6 + Math.round(6*ampBoost);         // 6..12
const decay     = 0.06 * (1.0 - 0.35*ampBoost);       // slower decay when loud
ctx.strokeStyle = 'rgba(255,216,77,0.30)';

for (let k=1; k<=echoCount; k++){
  ctx.beginPath();
  for (let s=0; s<=steps; s++){
    const x = R + (s/steps)*(2*orbR);

    // time shifted back -> echoes trail behind
    const ph = omega*(t - k*0.010*(1.0 - 0.4*ampBoost)) - ksp*(x - R);

    // echoes use simpler mix for clarity
    let y0 = (
      1.05*Math.sin(ph) +
      0.35*ampBoost*Math.sin(2*ph + 0.15) +
      0.20*ampBoost*Math.sin(3*ph + 0.75)
    );

    const y = cy + y0*(amp*(1.0 - decay*k));
    if (s===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}

ctx.restore();


  // --- Stats overlay (fps / fft max / amplitude / peak frequency) ---
  ctx.save();
  ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(`fps: ${fps.toFixed(2)}`, W - 95, H - 20);
  ctx.fillText(`fft max: ${fft_max.toFixed(2)}`, 25, H - 20);
  ctx.fillText(`peak frequency: ${peak_frequency.toFixed(2)}`, 25, H - 35);
  ctx.restore();
}
