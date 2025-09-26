import { useEffect, useRef, useState } from "react";
import type { MicAnalyser } from "../audio/useMicAnalyser";
import {
  detectPitchACF,
  simpleCepstrumPitch,
  predictNextMidi,
  midiToNoteName,
  midiToFreq,
} from "../audio/pitch";
import {
  Boid,
  BoidParams,
  makeBoids,
  stepBoids,
  drawBoids,
} from "../visuals/Boids";
import {
  Sparkle,
  burstSparkles,
  stepSparkles,
  drawSparkles,
} from "../visuals/Sparkles";

type Props = { analyser: MicAnalyser; className?: string };

type Swarm = {
  boids: Boid[];
  hueBase: number; // base hue that drifts
  life: number; // 0..1 fade
  alive: boolean;
  targetPhase: number; // to spread targets
};

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

const KS_MAJOR = [
  6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KS_MINOR = [
  6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

export default function VisualizerPro({ analyser, className }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [fps, setFps] = useState(0);
  const [debugOpen, setDebugOpen] = useState(false);
  const [combo, setCombo] = useState(true); // combine swarm + orb+rays
  const [mode, setMode] = useState<"SWARM" | "ORB" | "RAYS">("SWARM");

  // multi-swarm state
  const swarmsRef = useRef<Swarm[]>([]);
  const sparksRef = useRef<Sparkle[]>([]);
  const boidParams: BoidParams = {
    count: 120,
    maxSpeed: 3.6,
    align: 0.09,
    coh: 0.015,
    sep: 0.002,
    range: 60,
  };

  // audio buffers
  const fftRef = useRef<Float32Array | null>(null);
  const fftSmoothedRef = useRef<Float32Array | null>(null);
  const timeRef = useRef<Uint8Array | null>(null);
  const midiHist = useRef<number[]>([]);
  const lastBeat = useRef(0);
  const fluxPrev = useRef<Float32Array | null>(null);
  const frameMsAvg = useRef<number>(16.7);

  // BPM estimation buffers
  const fluxHist = useRef<number[]>([]);
  const bpmVal = useRef<number | null>(null);

  // prediction accuracy stats
  const predQueue = useRef<number[]>([]);
  const errorMAE = useRef(0);
  const errorCount = useRef(0);
  // NEW: additional MAE metrics
  const errorMAEHz = useRef(0);
  const errorCountHz = useRef(0);
  const errorMAECents = useRef(0);
  const errorCountCents = useRef(0);

  useEffect(() => {
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    let W = (c.width = c.clientWidth || window.innerWidth);
    let H = (c.height = c.clientHeight || window.innerHeight);

    const onResize = () => {
      W = c.width = c.clientWidth || window.innerWidth;
      H = c.height = c.clientHeight || window.innerHeight;
    };
    window.addEventListener("resize", onResize);

    function spawnSwarm(): Swarm {
      const hueBase = Math.random() * 360;
      const s: Swarm = {
        boids: makeBoids(
          W,
          H,
          Math.floor(boidParams.count * (0.6 + Math.random() * 0.8))
        ),
        hueBase,
        life: 0,
        alive: true,
        targetPhase: Math.random() * Math.PI * 2,
      };
      s.boids.forEach((b, i) => (b.hue = hueBase + (i % 32) * 1.7));
      return s;
    }
    swarmsRef.current = [];
    const initial = Math.floor(2 + Math.random() * 3);
    for (let i = 0; i < initial; i++) swarmsRef.current.push(spawnSwarm());

    let last = performance.now();
    let raf = 0;

    const loop = () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      setFps((prev) => lerp(prev, 1 / Math.max(1e-3, dt), 0.1));
      frameMsAvg.current = lerp(frameMsAvg.current, dt * 1000, 0.2);

      ctx.clearRect(0, 0, W, H);

      // pull audio
      const a = analyser.node();
      const sr = analyser.sampleRate() ?? 48000;
      if (a) {
        if (!fftRef.current)
          fftRef.current = new Float32Array(analyser.frequencyBinCount());
        if (!fftSmoothedRef.current)
          fftSmoothedRef.current = new Float32Array(
            analyser.frequencyBinCount()
          );
        if (!timeRef.current) timeRef.current = new Uint8Array(a.fftSize);
        analyser.getFloatFrequency(fftRef.current);
        analyser.getTimeDomain(timeRef.current);

        // smooth & normalize rays to avoid "cliff"
        const mag = fftRef.current!;
        const sm = fftSmoothedRef.current!;
        let localMax = 1e-6;
        for (let i = 0; i < mag.length; i++) {
          sm[i] = lerp(sm[i], Math.max(0, mag[i] + 140), 0.2); // smooth dB upshift
          if (sm[i] > localMax) localMax = sm[i];
        }
        for (let i = 0; i < sm.length; i++) sm[i] = sm[i] / localMax; // normalize 0..1
      }

      const mag = fftRef.current;
      const magN = fftSmoothedRef.current;
      const tdom = timeRef.current;

      // pitch detection + prediction
      let pitchHz: number | null = null,
        pitchMidi: number | null = null,
        pitchNote: string | null = null,
        clarity = 0;

      if (tdom) {
        const f32 = new Float32Array(tdom.length);
        for (let i = 0; i < tdom.length; i++) f32[i] = (tdom[i] - 128) / 128;
        const r1 = detectPitchACF(f32, sr);
        const r2 = mag
          ? simpleCepstrumPitch(mag, sr)
          : { frequency: null, midi: null, note: null, clarity: 0 };
        const best = r1.clarity >= r2.clarity ? r1 : r2;
        pitchHz = best.frequency;
        pitchMidi = best.midi;
        pitchNote = best.note;
        clarity = best.clarity;

        if (pitchMidi && clarity > 0.3) {
          midiHist.current.push(pitchMidi);
          if (midiHist.current.length > 128) midiHist.current.shift();
          if (predQueue.current.length) {
            const pred = predQueue.current.shift()!;
            // --- Semitone error (MIDI space)
            const errSemi = Math.abs(pred - pitchMidi);
            errorMAE.current =
              (errorMAE.current * errorCount.current + errSemi) /
              (errorCount.current + 1);
            errorCount.current += 1;

            // --- Frequency-space errors (Hz & cents)
            const predHz = midiToFreq(pred);
            if (pitchHz) {
              const errHz = Math.abs(predHz - pitchHz);
              errorMAEHz.current =
                (errorMAEHz.current * errorCountHz.current + errHz) /
                (errorCountHz.current + 1);
              errorCountHz.current += 1;

              // cents = 1200 * log2(actual/pred)
              const errCents = Math.abs(1200 * Math.log2(pitchHz / predHz));
              errorMAECents.current =
                (errorMAECents.current * errorCountCents.current + errCents) /
                (errorCountCents.current + 1);
              errorCountCents.current += 1;
            }
          }
        }
      }
      const predicted = predictNextMidi(midiHist.current);
      if (predicted != null) predQueue.current.push(predicted);

      // spectral flux + BPM
      let bassEnergy = 0,
        beat = false,
        flux = 0;
      if (mag) {
        const prev = fluxPrev.current ?? new Float32Array(mag.length);
        let sum = 0;
        for (let i = 0; i < mag.length; i++) {
          const v = Math.max(0, mag[i] + 140);
          const diff = Math.max(0, v - prev[i]);
          sum += diff;
          prev[i] = v;
        }
        flux = sum / mag.length;
        fluxPrev.current = prev;
        // BPM estimation via autocorrelation over flux history
        fluxHist.current.push(flux);
        if (fluxHist.current.length > 1024) fluxHist.current.shift();
        const bpm = estimateBPM(fluxHist.current, 60, 200, frameMsAvg.current); // acceptable range 60..200 BPM
        bpmVal.current = bpm;

        if (flux > 0.8) {
          const since = now - lastBeat.current;
          if (since > 120) {
            beat = true;
            lastBeat.current = now;
          }
        }
        // bass band 20-150Hz
        const bins = mag.length;
        const hzPerBin = sr / 2 / bins;
        const maxBin = Math.min(bins - 1, Math.floor(150 / hzPerBin));
        for (let i = 0; i <= maxBin; i++) {
          bassEnergy += Math.max(0, mag[i] + 140);
        }
        bassEnergy /= maxBin + 1;
      }

      // Key detection (very light-weight chroma with Krumhansl profiles)
      const keyText = mag ? estimateKeyFromFFT(mag, sr) : "—";

      // -------------------- DRAW ORB & RAYS --------------------
      const drawOrbAndRays = () => {
        const R = Math.min(W, H) * 0.26;
        // Slow, passionate pulse; blue/black palette
        const slow = 0.0018; // slower
        // Use Order-3 energy to modulate pulse more musically
        const order2_for_pulse = mag ? computeCepstrum(mag) : null;
        const order3_full = order2_for_pulse ? fftMag(order2_for_pulse) : null;
        let o3Energy = 0;
        if (order3_full) {
          // RMS-like energy
          let s = 0;
          for (let i = 0; i < order3_full.length; i++) {
            const v = order3_full[i];
            if (isFinite(v)) s += v * v;
          }
          o3Energy = Math.sqrt(s / Math.max(1, order3_full.length));
          // normalize roughly
          o3Energy = Math.min(1, o3Energy / 5000);
        }
        const orbPulse =
          0.9 + 0.12 * Math.sin(performance.now() * slow) + 0.08 * o3Energy;
        const orbR = R * orbPulse;

        // fixed blue palette
        const noteHue = ((predicted ?? 60) % 12) * (360 / 12);
        const baseBlue = 220;
        const raysHue = (baseBlue * 0.6 + noteHue * 0.4) % 360;
        const center = `hsla(${baseBlue}, 85%, 65%, 1)`;
        const mid = `hsla(220, 65%, 35%, 0.45)`;

        const grd = ctx.createRadialGradient(
          W / 2,
          H / 2,
          orbR * 0.2,
          W / 2,
          H / 2,
          orbR * 1.7
        );
        grd.addColorStop(0, center);
        grd.addColorStop(0.45, mid);
        grd.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, orbR * 1.6, 0, Math.PI * 2);
        ctx.fill();

        // rare strobe to accent downbeats only
        if (
          (bpmVal.current ?? 0) > 0 &&
          Math.sin(performance.now() * 0.02) > 0.98
        ) {
          ctx.strokeStyle = `hsla(220, 90%, 70%, 0.6)`;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.arc(W / 2, H / 2, orbR * 1.05, 0, Math.PI * 2);
          ctx.stroke();
        }

        // normalized rays continuous around circle
        if (magN) {
          const N = magN.length;
          ctx.save();
          ctx.translate(W / 2, H / 2);
          const base = orbR * 1.08;
          for (let k = 0; k < N; k++) {
            const val = clamp01(magN[k]); // 0..1 normalized
            const angle = (k / N) * Math.PI * 2 + performance.now() * 0.00025;
            const r1 = base;
            const r2 = base + Math.pow(val, 0.9) * 160;
            const c = `hsla(${raysHue}, 80%, ${30 + val * 35}%, ${
              0.35 + val * 0.35
            })`;
            ctx.strokeStyle = c;
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.moveTo(Math.cos(angle) * r1, Math.sin(angle) * r1);
            ctx.lineTo(Math.cos(angle) * r2, Math.sin(angle) * r2);
            ctx.stroke();
          }
          ctx.restore();
        }

        // --- Overlay centerline visuals inside the orb ---
        // Clip to orb circle so lines live "inside" the orb face
        ctx.save();
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, orbR * 1.05, 0, Math.PI * 2);
        ctx.clip();

        const chordW = orbR * 2; // diameter
        const leftX = W / 2 - chordW / 2;
        const rightX = W / 2 + chordW / 2;
        const centerY = H / 2;

        // 1) Waveform (yellow) across center
        if (tdom) {
          const wf = new Float32Array(tdom.length);
          for (let i = 0; i < tdom.length; i++) wf[i] = (tdom[i] - 128) / 128;
          // normalize locally
          let min = 1e9,
            max = -1e9;
          for (let i = 0; i < wf.length; i++) {
            const v = wf[i];
            if (v < min) min = v;
            if (v > max) max = v;
          }
          const range = max - min || 1;
          ctx.beginPath();
          for (let i = 0; i < 200; i++) {
            const idx = Math.floor((i / 200) * wf.length);
            const v = (wf[idx] - min) / range; // 0..1
            const y = centerY + (v - 0.5) * (orbR * 0.6); // ±30% radius
            const x = leftX + (i / 199) * (rightX - leftX);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.strokeStyle = "rgba(255,216,77,0.9)"; // original yellow with translucency
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }

        // 2) Order-3 FFT (non-halved) as white histogram + slanted tails from axis
        // if (order3_full){
        //   // normalize
        //   let m=1e9, M=-1e9;
        //   for(let i=0;i<order3_full.length;i++){ const v=order3_full[i]; if(!isFinite(v)) continue; if(v<m)m=v; if(v>M)M=v; }
        //   const rng = (M-m)||1;
        //   const bins = 260; // dense sampling across diameter
        //   const slope = 0.35; // slant factor slightly steeper
        //   // Histogram bars (semi-transparent) across the center
        //   for (let i=0;i<bins;i++){
        //     const idx = Math.floor(i/bins * order3_full.length);
        //     const nv = (order3_full[idx]-m)/rng; // 0..1
        //     const amp = nv * (orbR*0.6);
        //     const x = leftX + (i/(bins-1)) * (rightX-leftX);
        //     // main vertical bar centered on centerY
        //     const yTop = centerY - amp*0.15;
        //     const yBot = centerY + amp*0.15;
        //     ctx.beginPath();
        //     ctx.moveTo(x, yTop);
        //     ctx.lineTo(x, yBot);
        //     ctx.strokeStyle = "rgba(255,255,255,0.65)";
        //     ctx.lineWidth = 1.0;
        //     ctx.stroke();
        //     // slanted tail downward
        //     const x2 = x + amp * slope;
        //     const y2 = centerY + amp;
        //     ctx.beginPath();
        //     ctx.moveTo(x, centerY);
        //     ctx.lineTo(x2, y2);
        //     ctx.strokeStyle = "rgba(255,255,255,0.95)";
        //     ctx.lineWidth = 1.0;
        //     ctx.stroke();
        //   }
        // }
        ctx.restore();
      };

      // -------------------- DRAW SWARMS --------------------
      const drawSwarms = () => {
        // spawn/despawn
        if (
          swarmsRef.current.length < 2 ||
          (swarmsRef.current.length < 5 && Math.random() < 0.01)
        ) {
          swarmsRef.current.push(spawnSwarm());
        }
        if (swarmsRef.current.length > 2 && Math.random() < 0.004) {
          const idx = Math.floor(Math.random() * swarmsRef.current.length);
          swarmsRef.current[idx].alive = false;
        }

        const pred = predicted ?? 60;
        const thetaBase =
          (pred / 12) * Math.PI * 2 + performance.now() * 0.0005;

        const centers = swarmsRef.current.map((s) => centerOf(s.boids));
        for (let i = 0; i < swarmsRef.current.length; i++) {
          const s = swarmsRef.current[i];
          // drift but bias toward blue scheme
          s.hueBase = (s.hueBase * 0.95 + 220 * 0.05) % 360;
          s.life = clamp01(lerp(s.life, s.alive ? 1 : 0, 0.02));
          const radius =
            Math.min(W, H) * 0.25 +
            Math.sin(performance.now() * 0.001 + s.targetPhase) * 18;
          const target = {
            x: W / 2 + Math.cos(thetaBase + s.targetPhase) * radius,
            y: H / 2 + Math.sin(thetaBase + s.targetPhase) * radius,
          };

          let tx = target.x,
            ty = target.y;
          for (let j = 0; j < centers.length; j++) {
            if (i === j) continue;
            tx = lerp(tx, centers[j].x, 0.015);
            ty = lerp(ty, centers[j].y, 0.015);
          }

          stepBoids(
            s.boids,
            W,
            H,
            boidParams,
            { x: tx, y: ty },
            Math.min(2.5, (bassEnergy + (beat ? 25 : 0)) * 0.02)
          );

          // continuous sparkles
          if (Math.random() < 0.7 + o3Energy * 0.15) {
            const b = s.boids[Math.floor(Math.random() * s.boids.length)];
            burstSparkles(
              sparksRef.current,
              b.x,
              b.y,
              3,
              0.2 + Math.min(1.5, bassEnergy * 0.01)
            );
          }

          ctx.save();
          ctx.globalAlpha = 0.6 * s.life;
          drawBoids(
            ctx,
            s.boids.map((b) => {
              // converge hues toward blue-ish with gentle variance
              b.hue = (s.hueBase + b.hue * 0.08) % 360;
              return b;
            })
          );
          ctx.restore();
        }

        if (Math.random() < 0.2 + o3Energy * 0.1 || flux > 0.8 || beat) {
          burstSparkles(
            sparksRef.current,
            W / 2,
            H / 2,
            8,
            Math.min(2, bassEnergy * 0.02)
          );
        }
        stepSparkles(sparksRef.current, W, H);
        drawSparkles(ctx, sparksRef.current);
      };

      function centerOf(boids: Boid[]) {
        let x = 0,
          y = 0;
        for (const b of boids) {
          x += b.x;
          y += b.y;
        }
        const n = Math.max(1, boids.length);
        return { x: x / n, y: y / n };
      }

      // Pre-calc Order-3 energy for global dynamics (sparkles, etc.)
      let o3Energy = 0;
      if (mag) {
        const _o2 = computeCepstrum(mag);
        const _o3 = fftMag(_o2);
        let s = 0;
        for (let i = 0; i < _o3.length; i++) {
          const v = _o3[i];
          if (isFinite(v)) s += v * v;
        }
        o3Energy = Math.min(1, Math.sqrt(s / Math.max(1, _o3.length)) / 5000);
      }
      // draw combined or modes
      if (combo) {
        drawOrbAndRays();
        drawSwarms();
      } else {
        if (mode === "ORB" || mode === "RAYS") drawOrbAndRays();
        if (mode === "SWARM") drawSwarms();
      }

      // -------------------- DEBUG PANEL --------------------
      if (debugOpen) {
        const pad = 8,
          w = 240,
          h = 60;
        const x0 = 12,
          y0 = 120;

          // Space reserved for the debug text block (prevents overlap with mini-graphs)
const headerLines = 5;   // Pitch, Pred/MAE(semi), MAE(cents/Hz), Tempo/Key, FPS/Beat
const lineH = 16;        // 12px font ≈ 16px line height
const headerH = headerLines * lineH + pad * 2;  // top/bottom breathing room

        ctx.save();
        ctx.fillStyle = "rgba(8,11,31,0.88)";
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;

        // 6 rows: Waveform + Orders 1..5
        const rows = 6;
        const panelH = rows * h + (rows + 2) * pad + headerH;
        ctx.fillRect(x0 - 6, y0 - 50, w + 16, panelH);
        ctx.strokeRect(x0 - 6, y0 - 50, w + 16, panelH);

        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.font = "12px ui-sans-serif, system-ui";

        const noteText = pitchNote
          ? `${pitchNote}  ${pitchHz?.toFixed(1)}Hz  q=${clarity.toFixed(2)}`
          : "—";

        const maeSemi = errorCount.current ? errorMAE.current.toFixed(2) : "—";
        const maeHz = errorCountHz.current
          ? errorMAEHz.current.toFixed(2)
          : "—";
        const maeCent = errorCountCents.current
          ? errorMAECents.current.toFixed(1)
          : "—";
        const bpmText = bpmVal.current
          ? `${bpmVal.current.toFixed(1)} BPM`
          : "—";

        ctx.fillText(`Pitch: ${noteText}`, x0, y0 - 25);
        // line 1 (unchanged above this): ctx.fillText(`Pitch: ${noteText}`, x0, y0 - 25);

        // line 2: semi only
        ctx.fillText(
          `Pred Next: ${
            predicted != null ? midiToNoteName(predicted) : "—"
          }  MAE(semi): ${maeSemi}`,
          x0,
          y0 - 10
        );

        // line 3: new line for cents + Hz
        ctx.fillText(
          `MAE(cents): ${maeCent}  ·  MAE(Hz): ${maeHz}`,
          x0,
          y0 + 5
        );

        // push the following lines down by one row
        ctx.fillText(`Tempo: ${bpmText}  Key: ${keyText}`, x0, y0 + 20);
        ctx.fillText(
          `FPS ${fps.toFixed(0)}  Beat ${flux.toFixed(2)}`,
          x0,
          y0 + 35
        );

        // Mini-graphs
        const yBase = y0 + headerH;


        // Waveform (unfiltered time domain)
        const timeData = tdom
          ? new Float32Array(Array.from(tdom, (v) => (v - 128) / 128))
          : null;
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 0,
          w,
          h,
          timeData,
          "Waveform (time)",
          false,
          true
        );

        // Order 1 (FFT)
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 1,
          w,
          h,
          mag,
          "FFT (Order 1)",
          false,
          false
        );

        // Order 2 (Cepstrum)
        const order2 = mag ? computeCepstrum(mag) : null;
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 2,
          w,
          h,
          order2,
          "Order 2 (Cepstrum)",
          false,
          false
        );

        // Order 3 (FFT of Cepstrum) - half-spectrum expanded
        const order3 = order2 ? fftMag(order2) : null;
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 3,
          w,
          h,
          order3,
          "Order 3 (half, expanded)",
          true,
          false
        );

        // Order 4 (FFT of Order 3)
        const order4 = order3 ? fftMag(order3) : null;
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 4,
          w,
          h,
          order4,
          "Order 4",
          false,
          false
        );

        // Order 5 (FFT of Order 4) - half-spectrum expanded
        const order5 = order4 ? fftMag(order4) : null;
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 5,
          w,
          h,
          order5,
          "Order 5 (half, expanded)",
          true,
          false
        );

        ctx.restore();
      }

      // Helper for mini-graphs
      function drawMini(
        c: CanvasRenderingContext2D,
        x: number,
        y: number,
        w: number,
        h: number,
        data: Float32Array | null,
        label: string,
        half: boolean,
        isWave: boolean
      ) {
        c.save();
        c.strokeStyle = "rgba(255,255,255,0.15)";
        c.strokeRect(x - 2, y - 2, w + 4, h + 4);
        c.fillStyle = "rgba(255,255,255,0.08)";
        c.fillRect(x, y, w, h);
        c.fillStyle = "rgba(255,255,255,0.85)";
        c.font = "11px ui-sans-serif, system-ui";
        c.fillText(label, x + 6, y + 14);

        if (data && data.length) {
          let arr = data;
          if (half) {
            const N = Math.max(1, Math.floor(arr.length / 2));
            arr = arr.slice(0, N);
          }
          let min = Infinity,
            max = -Infinity;
          for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (isFinite(v)) {
              if (v < min) min = v;
              if (v > max) max = v;
            }
          }
          const range = max - min || 1;

          c.beginPath();
          for (let i = 0; i < w; i++) {
            const idx = Math.floor((i / w) * arr.length);
            const nv = (arr[idx] - min) / range;
            const yy = y + h - nv * h;
            const xx = x + i;
            if (i === 0) c.moveTo(xx, yy);
            else c.lineTo(xx, yy);
          }
          c.strokeStyle = isWave
            ? "rgba(160,200,255,0.95)"
            : "rgba(255,240,200,0.9)";
          c.lineWidth = 1;
          c.stroke();
        }
        c.restore();
      }
    };

    function drawMini(
      c: CanvasRenderingContext2D,
      x: number,
      y: number,
      w: number,
      h: number,
      data: Float32Array | null,
      label: string,
      half: boolean,
      isWave: boolean
    ) {
      c.save();
      c.strokeStyle = "rgba(255,255,255,0.15)";
      c.strokeRect(x - 2, y - 2, w + 4, h + 4);
      c.fillStyle = "rgba(255,255,255,0.08)";
      c.fillRect(x, y, w, h);
      c.fillStyle = "rgba(255,255,255,0.85)";
      c.font = "11px ui-sans-serif, system-ui";
      c.fillText(label, x + 6, y + 14);
      if (data) {
        let arr = data;
        // Use first half for mirrored spectra
        if (half) {
          const N = Math.floor(arr.length / 2);
          arr = arr.slice(0, Math.max(1, N));
        }
        // normalize
        let min = Infinity,
          max = -Infinity;
        for (let i = 0; i < arr.length; i++) {
          const v = arr[i];
          if (isFinite(v)) {
            if (v < min) min = v;
            if (v > max) max = v;
          }
        }
        const range = max - min || 1;
        c.beginPath();
        for (let i = 0; i < w; i++) {
          const idx = Math.floor((i / w) * arr.length);
          const nv = (arr[idx] - min) / range;
          const yy = y + h - nv * h;
          const xx = x + i;
          if (i === 0) c.moveTo(xx, yy);
          else c.lineTo(xx, yy);
        }
        c.strokeStyle = isWave
          ? "rgba(160,200,255,0.95)"
          : "rgba(255,240,200,0.9)";
        c.lineWidth = 1;
        c.stroke();
      }
      c.restore();
    }

    function computeCepstrum(mag: Float32Array): Float32Array {
      const N = mag.length;
      const logMag = new Float32Array(N);
      for (let i = 0; i < N; i++)
        logMag[i] = Math.log(1e-12 + Math.max(0, mag[i] + 140));
      const cep = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++)
          sum += logMag[n] * Math.cos((2 * Math.PI * k * n) / N);
        cep[k] = sum;
      }
      return cep;
    }
    function fftMag(signal: Float32Array): Float32Array {
      const N = signal.length;
      const out = new Float32Array(N);
      for (let k = 0; k < N; k++) {
        let re = 0,
          im = 0;
        for (let n = 0; n < N; n++) {
          const ang = (-2 * Math.PI * k * n) / N;
          re += signal[n] * Math.cos(ang);
          im += signal[n] * Math.sin(ang);
        }
        out[k] = Math.hypot(re, im);
      }
      return out;
    }

    function estimateBPM(
      series: number[],
      minBPM: number,
      maxBPM: number,
      frameMs: number
    ): number | null {
      if (series.length < 128) return null;
      // autocorrelation
      const N = series.length;
      const mean = series.reduce((a, b) => a + b, 0) / N;
      const x = series.map((v) => v - mean);
      const ac: number[] = [];
      for (let lag = 1; lag < N / 2; lag++) {
        let sum = 0;
        for (let i = 0; i < N - lag; i++) sum += x[i] * x[i + lag];
        ac[lag] = sum;
      }
      // search lags corresponding to bpm range
      let bestLag = -1,
        bestVal = -Infinity;
      for (let lag = 8; lag < ac.length; lag++) {
        const periodMs = lag * frameMs;
        const bpm = 60000 / Math.max(1, periodMs);
        if (bpm >= minBPM && bpm <= maxBPM) {
          if (ac[lag] > bestVal) {
            bestVal = ac[lag];
            bestLag = lag;
          }
        }
      }
      if (bestLag < 0) return null;
      const periodMs = bestLag * frameMs;
      return 60000 / Math.max(1, periodMs);
    }

    function estimateKeyFromFFT(mag: Float32Array, sampleRate: number): string {
      // crude chroma by bin mapping
      const N = mag.length;
      const bins = Array(12).fill(0);
      for (let k = 1; k < N; k++) {
        const freq = (k * (sampleRate / 2)) / N;
        if (freq < 50 || freq > 5000) continue;
        const midi = 69 + 12 * Math.log2(freq / 440);
        const cls = ((Math.round(midi) % 12) + 12) % 12;
        const power = Math.max(0, mag[k] + 140);
        bins[cls] += power;
      }
      // rotate and score
      function score(profile: number[]): { key: number; val: number } {
        let bestK = 0,
          bestV = -Infinity;
        for (let shift = 0; shift < 12; shift++) {
          let s = 0;
          for (let i = 0; i < 12; i++) s += bins[(i + shift) % 12] * profile[i];
          if (s > bestV) {
            bestV = s;
            bestK = shift;
          }
        }
        return { key: bestK, val: bestV };
      }
      const maj = score(KS_MAJOR);
      const minr = score(KS_MINOR);
      const names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
      ];
      if (maj.val >= minr.val) return names[maj.key] + " major";
      return names[minr.key] + " minor";
    }

    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, [analyser, combo, mode, debugOpen]);

  return (
    <div
      className={className ?? ""}
      style={{ position: "relative", width: "100%", height: "100%" }}
    >
      <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
      <div className="mode-toggle">
        <button
          className={combo ? "active" : ""}
          onClick={() => setCombo(!combo)}
        >
          {combo ? "Combine" : "Separate"}
        </button>
        <button
          className={mode === "SWARM" ? "active" : ""}
          onClick={() => setMode("SWARM")}
        >
          Swarm
        </button>
        <button
          className={mode === "RAYS" ? "active" : ""}
          onClick={() => setMode("RAYS")}
        >
          Orb+Rays
        </button>
        <button
          className={mode === "ORB" ? "active" : ""}
          onClick={() => setMode("ORB")}
        >
          Orb Only
        </button>
        <button onClick={() => setDebugOpen(!debugOpen)}>
          {debugOpen ? "Hide Debug" : "Show Debug"}
        </button>
      </div>
    </div>
  );
}
