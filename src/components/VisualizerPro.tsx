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
import { useWakeLock } from "../hooks/useWakeLock"; // adjust the path if needed

type Props = { analyser: MicAnalyser; className?: string };

type Swarm = {
  boids: Boid[];
  hueBase: number; // base hue that drifts
  life: number; // 0..1 fade
  alive: boolean;
  targetPhase: number; // to spread targets
};

// ---- Global debug/state snapshot (updated every frame) ----
export const VIS = {
  // timing
  fps: 0,
  frameMs: 16.7,

  // audio/pitch
  pitchHz: null as number | null,
  pitchMidi: null as number | null,
  pitchNote: null as string | null,
  clarity: 0,
  amplitude: 0,

  // prediction & error stats
  predictedMidi: null as number | null,
  maeSemi: 0,
  maeHz: 0,
  maeCents: 0,
  errCountSemi: 0,
  errCountHz: 0,
  errCountCents: 0,

  // beat/tempo/energy
  flux: 0,
  bassEnergy: 0,
  bpm: null as number | null,

  // key
  keyText: "—",

  // snapshots for debug minis (reuse same references each frame)
  timeData: null as Float32Array | null, // waveform ([-1..1])
  fftOrder1: null as Float32Array | null, // original mag (Float freq dB+140)
  order2: null as Float32Array | null, // cepstrum
  order3: null as Float32Array | null, // fft(order2)
  order4: null as Float32Array | null, // fft(order3)
  order5: null as Float32Array | null, // fft(order4)
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
  const toolbarStyle: React.CSSProperties = {
    position: "absolute",
    right: 16,
    top: 16,
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: 8,
    padding: "10px 12px",
    background: "rgba(8,11,31,0.72)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 14,
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    zIndex: 1000,
  };

  const btnBase: React.CSSProperties = {
    // reset any external styles
    all: "unset" as any,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "8px 12px",
    borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.18)",
    background: "rgba(22,26,46,0.85)",
    color: "#fff",
    fontSize: 13,
    fontWeight: 600,
    letterSpacing: 0.2 as any,
    cursor: "pointer",
    lineHeight: 1,
    userSelect: "none",
  };

  const btnActive: React.CSSProperties = {
    ...btnBase,
    background: "rgba(255,215,100,0.15)",
    border: "1px solid rgba(255,215,100,0.6)",
    color: "rgb(255,215,100)",
  };

  const SPEED_ORDER = 100.0; // nonlinear mapping order for speed
  const POP_COOLDOWN_MS = 100; // cooldown for pop effect

  // --- Swarm reactivity state (clap, smoothing, shake) ---
  const AMP_BUF = 12; // frames to smooth amplitude
  const CLAP_COOLDOWN_MS = 20;
  const CLAP_MIN = 0.4; // minimum amplitude to even consider a clap
  const CLAP_DELTA = 0.15; // how “spiky” it must be vs previous

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [fps, setFps] = useState(0);
  const [debugOpen, setDebugOpen] = useState(false);
  const [combo, setCombo] = useState(true); // combine swarm + orb+rays
  const [mode, setMode] = useState<"SWARM" | "ORB" | "RAYS">("SWARM");
  // Track whether the mic/analyser is currently active (driven by AudioContext state)
  const [micActive, setMicActive] = useState(true);

  // Keep the screen awake while this component is mounted/active
  useWakeLock(true);

  // dynamics state for the new swarm logic
  const orderEnergyPrevRef = useRef({ e1: 0, e2: 0, e3: 0, e4: 0, e5: 0 });
  const lastSwarmPopRef = useRef(0);
  const ampBufRef = useRef<number[]>([]);
  const ampPrevRef = useRef(0);
  const clapUntilRef = useRef(0);
  const shakeRef = useRef(0); // 0..1 decays after clap
  const swirlEnergyRef = useRef(0); // 0..1, grows with sustained loudness

  // multi-swarm state
  const swarmsRef = useRef<Swarm[]>([]);
  const sparksRef = useRef<Sparkle[]>([]);
  // --- Rays temporal smoothing config/refs ---
  const RAY_AVG_FRAMES = 12; // <-- tweak this to smooth more/less
  const raysAvgBuf = useRef<Float32Array[]>([]); // ring buffer of recent frames
  const raysAvgTmp = useRef<Float32Array | null>(null); // scratch for averaged spectrum

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
  // prevent auto-resume when the user pressed Stop
  const userForcedOffRef = useRef(false);

  const boidParams: BoidParams = {
    count: 240,
    maxSpeed: 5.6,
    align: 0.19,
    coh: 0.15,
    sep: 0.02,
    range: 160,
  };

  // ----------
  // Audio control helpers (don’t rely on non-existent methods on analyser)
  // ----------
  // ---------- Audio helpers (with stream liveness checks) ----------

  async function stopMicSafely() {
  try {
    const node = analyser?.node?.() as any;
    const ctx: AudioContext | undefined = node?.context;

    // Suspend the graph (instant silence)
    if (ctx && ctx.state !== "suspended") {
      try { await ctx.suspend(); } catch {}
    }

    // Stop tracks so OS mic indicator turns off
    const stream = getCurrentStream();
    if (stream) {
      for (const t of stream.getTracks()) {
        try { t.stop(); } catch {}
      }
    }

    // Disconnect upstream source from analyser (we'll rebuild on resume)
    try { (analyser as any)?.source?.disconnect?.(node); } catch {}
    try { node?.disconnect?.(); } catch {}
  } finally {
    setMicActive(false);
  }
}

  function getAudioContext(): AudioContext | undefined {
    const node = analyser?.node?.();
    return (node && (node as any).context) as AudioContext | undefined;
  }

  function getCurrentStream(): MediaStream | undefined {
    const node = analyser?.node?.() as any;
    // Try several likely places where the stream might live
    return (
      (analyser as any)?.mediaStream ||
      (analyser as any)?.getStream?.() ||
      (analyser as any)?.source?.mediaStream || // MediaStreamSourceNode
      node?.mediaStream || // some wrappers attach it here
      (analyser as any)?.input?.mediaStream || // other common field names
      undefined
    );
  }

  function streamAlive(stream?: MediaStream): boolean {
    if (!stream) return false;
    const tracks = stream.getAudioTracks();
    return (
      tracks.length > 0 &&
      tracks.some((t) => t.readyState === "live" && !t.muted)
    );
  }

  async function suspendAudio() {
    try {
      const ctx = getAudioContext();
      if (ctx && ctx.state !== "suspended") await ctx.suspend();
    } catch {}
    setMicActive(false);
  }

  /**
   * Try to resume the graph. If the mic tracks were killed while backgrounded,
   * we try to re-init the analyser if it exposes an initializer, otherwise we
   * leave the button in "Start Mic" state for the user to click.
   */
  /** Return true if we end up with a running context + live stream. */
  async function resumeAudio(): Promise<boolean> {
    let ok = false;
    try {
      const node = analyser?.node?.() as any;
      const ctx: AudioContext | undefined = node?.context;
      if (!ctx) {
        setMicActive(false);
        return false;
      }

      // 1) Ensure context is running (must be from a user gesture)
      if (ctx.state !== "running") {
        await ctx.resume();
      }

      // 2) Try library hooks to resurrect the stream
      let stream = getCurrentStream();
      if (!streamAlive(stream)) {
        try {
          await (analyser as any)?.init?.();
        } catch {}
        try {
          await (analyser as any)?.start?.();
        } catch {}
        stream = getCurrentStream();
      }

      // 3) Hard fallback: ask for a brand new mic stream
      if (!streamAlive(stream)) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          (analyser as any).mediaStream = stream; // stash so getCurrentStream() can see it
        } catch (err) {
          // user denied or device missing
          setMicActive(false);
          return false;
        }
      }

      // 4) (Re)create and connect a MediaStreamSource feeding your analyser node
      let src = (analyser as any)?.source as
        | MediaStreamAudioSourceNode
        | undefined;
      // if no source or it points to a dead stream, rebuild it
      if (!src || (src as any).mediaStream !== stream) {
        try {
          src?.disconnect?.(node);
        } catch {}
        src = ctx.createMediaStreamSource(stream!);
        (analyser as any).source = src;
        try {
          src.connect(node);
        } catch {}
      }

      ok = ctx.state === "running" && streamAlive(stream);
    } catch {
      ok = false;
    }

    setMicActive(ok);
    return ok;
  }

  const togglingRef = { current: false }; // tiny debounce so rapid clicks don't race

  async function toggleMic() {
    if (togglingRef.current) return;
    togglingRef.current = true;
    try {
      const ctx = getAudioContext();
      const isRunning = !!ctx && ctx.state !== "suspended"; // ← no stream check here

      if (isRunning) {
        // User explicitly turned it off; don’t auto-resume on visibility change
        userForcedOffRef.current = true;
        await stopMicSafely();
      } else {
        userForcedOffRef.current = false;
        await resumeAudio();
      }
    } finally {
      togglingRef.current = false;
    }
  }

  // Pause when tab hidden / window blurred; resume when focused again
  useEffect(() => {
    const onHidden = () => suspendAudio();
    const onShown = () => {
      if (!userForcedOffRef.current) {
        void resumeAudio();
      } else setMicActive(false);
    };

    const onVisibility = () =>
      document.visibilityState === "visible" ? onShown() : onHidden();

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onHidden);
    window.addEventListener("focus", onShown);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onHidden);
      window.removeEventListener("focus", onShown);
    };
  }, []);

  useEffect(() => {
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    let W = (c.width = c.clientWidth || window.innerWidth);
    let H = (c.height = c.clientHeight || window.innerHeight);

    lastSwarmPopRef.current = performance.now() - POP_COOLDOWN_MS; // so first spike can happen
    orderEnergyPrevRef.current = { e1: 0, e2: 0, e3: 0, e4: 0, e5: 0 };

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

    // seed a few swarms
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

      // mirror into global
      VIS.fps = lerp(VIS.fps, 1 / Math.max(1e-3, dt), 0.1);
      VIS.frameMs = frameMsAvg.current;

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

      // Store for debug panels
      VIS.fftOrder1 = mag ?? null;
      VIS.timeData = tdom
        ? new Float32Array(Array.from(tdom, (v) => (v - 128) / 128))
        : null;

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

              // mirror into VIS
              VIS.maeSemi = errorMAE.current;
              VIS.errCountSemi = errorCount.current;
              VIS.maeHz = errorMAEHz.current;
              VIS.errCountHz = errorCountHz.current;
              VIS.maeCents = errorMAECents.current;
              VIS.errCountCents = errorCountCents.current;
            }
          }
        }
      }

      VIS.pitchHz = pitchHz;
      VIS.pitchMidi = pitchMidi;
      VIS.pitchNote = pitchNote;
      VIS.clarity = clarity;

      const predicted = predictNextMidi(midiHist.current);
      if (predicted != null) predQueue.current.push(predicted);
      VIS.predictedMidi = predicted ?? null;

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
        const bpm = estimateBPM(fluxHist.current, 60, 200, frameMsAvg.current);
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

        VIS.flux = flux;
        VIS.bpm = bpmVal.current;
        VIS.bassEnergy = bassEnergy;
      }

      // Key detection (very light-weight chroma with Krumhansl profiles)
      const keyText = mag ? estimateKeyFromFFT(mag, sr) : "—";
      VIS.keyText = keyText;

      // -------------------- DRAW ORB & RAYS --------------------
      const drawOrbAndRays = () => {
        const R = Math.min(W, H) * 0.26;
        const slow = 0.003; // slower
        const order2_for_pulse = mag ? computeCepstrum(mag) : null;
        const order3_full = order2_for_pulse ? fftMag(order2_for_pulse) : null;

        // expose orders globally
        VIS.order2 = order2_for_pulse;
        const order2 = VIS.order2;
        const order3 = order2 ? fftMag(order2) : null;
        VIS.order3 = order3;
        const order4 = order3 ? fftMag(order3) : null;
        VIS.order4 = order4;
        const order5 = order4 ? fftMag(order4) : null;
        VIS.order5 = order5;

        let o3Energy = 0;
        if (order3_full) {
          let s = 0;
          for (let i = 0; i < order3_full.length; i++) {
            const v = order3_full[i];
            if (isFinite(v)) s += v * v;
          }
          o3Energy = Math.sqrt(s / Math.max(1, order3_full.length));
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
          ctx.strokeStyle = `hsla(220, 90%, 70%, 0.85)`;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.arc(W / 2, H / 2, orbR * 1.05, 0, Math.PI * 2);
          ctx.stroke();
        }

        // ---- Rays around the orb using *smoothed* Order-2 (cepstrum) values ----
        const fftOffset = 100;
        if (order2_for_pulse) {
          const N = order2_for_pulse.length - 2 * fftOffset;

          if (!raysAvgTmp.current || raysAvgTmp.current.length !== N) {
            raysAvgTmp.current = new Float32Array(N);
            raysAvgBuf.current = [];
          }

          const cur = new Float32Array(N);
          for (let k = 0; k < N; k++) {
            cur[k] = Math.max(0, order2_for_pulse[k + fftOffset] / 160.0);
          }

          raysAvgBuf.current.push(cur);
          while (raysAvgBuf.current.length > RAY_AVG_FRAMES)
            raysAvgBuf.current.shift();

          const acc = raysAvgTmp.current!;
          acc.fill(0);
          for (const frame of raysAvgBuf.current) {
            for (let i = 0; i < N; i++) acc[i] += frame[i];
          }
          const denom = raysAvgBuf.current.length || 1;
          for (let i = 0; i < N; i++) acc[i] /= denom;

          ctx.save();
          ctx.translate(W / 2, H / 2);
          const base = orbR * 1.08;

          for (let k = 0; k < N; k++) {
            const val = acc[k]; // smoothed 0..~1
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
        ctx.save();
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, orbR * 1.05, 0, Math.PI * 2);
        ctx.clip();

        const chordW = orbR * 2; // diameter
        const leftX = W / 2 - chordW / 2;
        const rightX = W / 2 + chordW / 2;
        const centerY = H / 2;

        // 1) Waveform (yellow) across center, DC-centered
        if (tdom) {
          const wf = new Float32Array(tdom.length);
          let sum = 0;
          for (let i = 0; i < tdom.length; i++) {
            const v = (tdom[i] - 128) / 128;
            wf[i] = v;
            sum += v;
          }
          const mean = sum / wf.length;
          const orbAmp = 3.0;

          ctx.beginPath();
          const amp = orbR * orbAmp;
          const N = 200;
          for (let i = 0; i < N; i++) {
            const idx = Math.floor((i / (N - 1)) * (wf.length - 1));
            const v = wf[idx] - mean;
            const y = centerY + v * amp;
            const x = leftX + (i / (N - 1)) * (rightX - leftX);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.strokeStyle = "rgba(255,216,77,0.8)";
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }

        // 2) White bars across the center from order-2 (normalized)
        const whiteOffsetX = 20;
        if (order2_for_pulse) {
          let maxV = 1e-6;
          for (let i = whiteOffsetX; i < order2_for_pulse.length - 2 * whiteOffsetX; i++) {
            const v = order2_for_pulse[i];
            if (isFinite(v) && v > maxV) maxV = v;
          }

          const spectrum = new Float32Array(
            order2_for_pulse.length - whiteOffsetX
          );
          for (let i = whiteOffsetX; i < order2_for_pulse.length; i++) {
            const v = Math.max(0, order2_for_pulse[i]);
            spectrum[i - whiteOffsetX] = v / (maxV * 4);
          }

          const cx = W / 2,
            cy = H / 2;
          ctx.save();
          ctx.beginPath();
          ctx.arc(cx, cy, orbR * 0.96, 0, Math.PI * 2);
          ctx.clip();

          const bars = spectrum.length;
          const stride = Math.max(1, Math.floor(spectrum.length / bars));
          const innerH = orbR * 1.6;
          const left = cx - orbR * 0.9;
          const right = cx + orbR * 0.9;
          const width = right - left;
          const barW = width / bars;

          ctx.globalCompositeOperation = "lighter";
          for (let bi = 0, i = 0; bi < bars; bi++, i += stride) {
            const u = clamp01(spectrum[i] * 2.0);
            const h = u * innerH;
            const x = left + bi * barW;
            const y = cy - h * 0.5;
            ctx.fillStyle = `rgba(255,255,255,${0.35 * u})`;
            ctx.fillRect(x, y, Math.max(2, barW * 0.8), h);
          }
          ctx.globalCompositeOperation = "source-over";
          ctx.restore();
        }

        ctx.restore();
      };

      // -------------------- DRAW SWARMS --------------------
      const drawSwarms = () => {
        const nowMs = performance.now();
        const now = nowMs;

        // amplitude smoothing
        const amp = Math.max(
          0,
          Math.min(1, (globalThis as any).VIS?.amplitude ?? 0)
        );
        ampBufRef.current.push(amp);
        if (ampBufRef.current.length > AMP_BUF) ampBufRef.current.shift();

        let ampMA = 0;
        for (let i = 0; i < ampBufRef.current.length; i++)
          ampMA += ampBufRef.current[i];
        ampMA /= Math.max(1, ampBufRef.current.length);

        // clap detection
        const prev = ampPrevRef.current;
        const spike = amp - prev;
        const canClap = now > clapUntilRef.current;
        const isClap = canClap && amp > CLAP_MIN && spike > CLAP_DELTA;
        if (isClap) {
          clapUntilRef.current = now + CLAP_COOLDOWN_MS;
          shakeRef.current = 1; // max shake
          if (swarmsRef.current.length < 6) {
            // spawn burst
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
            swarmsRef.current.push(s);
          }
          burstSparkles(sparksRef.current, W / 2, H / 2, 16, 1.8);
          swarmsRef.current.forEach(
            (s) => (s.hueBase = (s.hueBase + 80) % 360)
          );
        }
        ampPrevRef.current = amp;

        // shake decay
        if (shakeRef.current > 0) {
          shakeRef.current = Math.max(
            0,
            shakeRef.current - 0.04 - 0.25 * (1 / Math.max(30, fps))
          );
        }

        // swirl energy
        const targetSwirl = ampMA; // 0..1
        swirlEnergyRef.current = lerp(
          swirlEnergyRef.current,
          targetSwirl,
          0.06
        );

        // baseline spawn/despawn
        if (
          swarmsRef.current.length < 2 ||
          (swarmsRef.current.length < 5 && Math.random() < 0.006)
        ) {
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
          swarmsRef.current.push(s);
        }
        if (swarmsRef.current.length > 2 && Math.random() < 0.003) {
          const idx = Math.floor(Math.random() * swarmsRef.current.length);
          swarmsRef.current[idx].alive = false;
        }

        const pred = predicted ?? 60;
        const thetaBase =
          (pred / 12) * Math.PI * 2 + performance.now() * 0.0005;

        // speed mapping with nonlinearity
        const speedScale =
          0.15 +
          Math.pow(ampMA, SPEED_ORDER) * 3.0 +
          (shakeRef.current > 0 ? 0.8 : 0);

        const orderKick = Math.min(1, (globalThis as any).VIS?.o3Energy ?? 0);
        const hueDriftRate = 35 + 120 * (ampMA * ampMA) + 160 * orderKick;

        const shake = shakeRef.current;
        const jitterMag = shake * (Math.min(W, H) * 0.02);
        const jitterX = (Math.random() * 2 - 1) * jitterMag;
        const jitterY = (Math.random() * 2 - 1) * jitterMag;

        const centers = swarmsRef.current.map((s) => centerOf(s.boids));

        for (let i = 0; i < swarmsRef.current.length; i++) {
          const s = swarmsRef.current[i];

          // rainbow drift
          s.hueBase = (s.hueBase + hueDriftRate * (1 / 60)) % 360;
          s.life = clamp01(lerp(s.life, s.alive ? 1 : 0, 0.02));

          const baseR = Math.min(W, H) * 0.25;
          const radius =
            baseR + Math.sin(performance.now() * 0.001 + s.targetPhase) * 18;

          const orbitAngle =
            thetaBase +
            s.targetPhase +
            swirlEnergyRef.current * 0.7 * Math.sin(performance.now() * 0.002);
          const target = {
            x: W / 2 + Math.cos(orbitAngle) * radius + jitterX,
            y: H / 2 + Math.sin(orbitAngle) * radius + jitterY,
          };

          let tx = target.x,
            ty = target.y;
          for (let j = 0; j < centers.length; j++) {
            if (i === j) continue;
            tx = lerp(tx, centers[j].x, 0.012);
            ty = lerp(ty, centers[j].y, 0.012);
          }

          if (swirlEnergyRef.current > 0.02) {
            const toC = { x: tx - W / 2, y: ty - H / 2 };
            const perp = { x: -toC.y, y: toC.x };
            const m = Math.hypot(perp.x, perp.y) || 1;
            const swirlPush =
              (0.002 + 0.02 * swirlEnergyRef.current) * Math.min(W, H);
            tx += (perp.x / m) * swirlPush;
            ty += (perp.y / m) * swirlPush;
          }

          const drive = Math.min(
            4.0,
            speedScale +
              Math.min(1.8, (VIS.bassEnergy + (VIS.flux > 0.8 ? 25 : 0)) * 0.02)
          );

          stepBoids(s.boids, W, H, boidParams, { x: tx, y: ty }, drive);

          if (isClap || Math.random() < 0.08 * ampMA + 0.02 * orderKick) {
            const c = centers[i] ?? { x: W / 2, y: H / 2 };
            burstSparkles(
              sparksRef.current,
              c.x + (Math.random() - 0.5) * 30,
              c.y + (Math.random() - 0.5) * 30,
              4 + Math.floor(6 * ampMA),
              0.4 + 1.4 * ampMA
            );
          }

          ctx.save();
          ctx.globalAlpha = 0.65 * s.life;
          drawBoids(
            ctx,
            s.boids.map((b, bi) => {
              const rainbowHue = (s.hueBase + bi * 1.2) % 360;
              const sat = 80 + 12 * ampMA;
              const light = 50 + 12 * ampMA;

              b.hue = rainbowHue;
              (b as any).bodyColor = `hsl(${rainbowHue}, ${sat}%, ${light}%)`;
              (b as any).trailColor = `hsla(${
                (rainbowHue + 25) % 360
              }, ${Math.min(95, sat + 4)}%, ${Math.max(35, light - 10)}%, ${
                0.65 + 0.25 * ampMA
              })`;
              return b;
            })
          );
          ctx.restore();
        }

        if (Math.random() < 0.08 + 0.25 * ampMA || isClap || VIS.flux > 0.8) {
          burstSparkles(
            sparksRef.current,
            W / 2,
            H / 2,
            6 + Math.floor(6 * ampMA),
            0.6 + 1.2 * ampMA
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
          y0 = 80;

        const headerLines = 5;
        const lineH = 16;
        const headerH = headerLines * lineH + pad * 2;

        ctx.save();
        ctx.fillStyle = "rgba(8,11,31,0.88)";
        ctx.strokeStyle = "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;

        const rows = 6;

        const panelTop = y0 - 25 - pad;
        const panelBottom = y0 + headerH + rows * h + (rows - 1) * pad + pad;
        const panelH = panelBottom - panelTop;

        ctx.fillRect(x0 - 6, panelTop, w + 16, panelH);
        ctx.strokeRect(x0 - 6, panelTop, w + 16, panelH);

        const yBase = y0 + headerH;

        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.font = "12px ui-sans-serif, system-ui";

        const noteText = VIS.pitchNote
          ? `${VIS.pitchNote}  ${VIS.pitchHz?.toFixed(
              1
            )}Hz  q=${VIS.clarity.toFixed(2)}`
          : "—";

        const maeSemi = VIS.errCountSemi ? VIS.maeSemi.toFixed(2) : "—";
        const maeHz = VIS.errCountHz ? VIS.maeHz.toFixed(2) : "—";
        const maeCent = VIS.errCountCents ? VIS.maeCents.toFixed(1) : "—";
        const bpmText = VIS.bpm ? `${VIS.bpm.toFixed(1)} BPM` : "—";

        ctx.fillText(`Pitch: ${noteText}`, x0, y0 - 25);
        ctx.fillText(
          `Pred Next: ${
            VIS.predictedMidi != null ? midiToNoteName(VIS.predictedMidi) : "—"
          }  MAE(semi): ${maeSemi}`,
          x0,
          y0 - 10
        );
        ctx.fillText(
          `MAE(cents): ${maeCent}  ·  MAE(Hz): ${maeHz}`,
          x0,
          y0 + 5
        );
        ctx.fillText(`Tempo: ${bpmText}  Key: ${VIS.keyText}`, x0, y0 + 20);
        ctx.fillText(
          `FPS ${VIS.fps.toFixed(0)}  Beat ${VIS.flux.toFixed(2)}`,
          x0,
          y0 + 35
        );

        // Waveform (unfiltered time domain) + amplitude capture
        const timeData = tdom
          ? new Float32Array(Array.from(tdom, (v) => (v - 128) / 128))
          : null;

        // Compute maximum absolute amplitude and mirror to VIS
        let timeMax = 0;
        if (timeData) {
          for (let i = 0; i < timeData.length; i++) {
            const v = Math.abs(timeData[i]);
            if (v > timeMax) timeMax = v;
          }
        }
        VIS.amplitude = timeMax;
        ctx.fillText(`Amplitude ${VIS.amplitude.toFixed(4)}`, x0, y0 + 50);

        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 0,
          w,
          h,
          VIS.timeData,
          "Waveform (time)",
          false,
          true
        );
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 1,
          w,
          h,
          VIS.fftOrder1,
          "FFT (Order 1)",
          false,
          false
        );

        const order2 = VIS.order2;
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

        const order3 = VIS.order3 ?? (order2 ? fftMag(order2) : null);
        const order4 = VIS.order4 ?? (order3 ? fftMag(order3) : null);
        const order5 = VIS.order5 ?? (order4 ? fftMag(order4) : null);

        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 3,
          w,
          h,
          order3,
          "Order 3",
          false,
          false
        );
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
        drawMini(
          ctx,
          x0,
          yBase + (h + pad) * 5,
          w,
          h,
          order5,
          "Order 5",
          false,
          false
        );
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
        if (half) {
          const N = Math.floor(arr.length / 2);
          arr = arr.slice(0, Math.max(1, N));
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
      const N = series.length;
      const mean = series.reduce((a, b) => a + b, 0) / N;
      const x = series.map((v) => v - mean);
      const ac: number[] = [];
      for (let lag = 1; lag < N / 2; lag++) {
        let sum = 0;
        for (let i = 0; i < N - lag; i++) sum += x[i] * x[i + lag];
        ac[lag] = sum;
      }
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
      <div style={toolbarStyle}>
        {/* <button
          style={combo ? btnActive : btnBase}
          onClick={() => setCombo(!combo)}
        >
          {combo ? "Combine" : "Separate"}
        </button>

        <button
          style={mode === "SWARM" ? btnActive : btnBase}
          onClick={() => setMode("SWARM")}
        >
          Swarm
        </button>

        <button
          style={mode === "ORB" ? btnActive : btnBase}
          onClick={() => setMode("ORB")}
        >
          Orb Only
        </button> */}

        <button
          style={debugOpen ? btnActive : btnBase}
          onClick={() => setDebugOpen(!debugOpen)}
        >
          {debugOpen ? "Hide Debug" : "Show Debug"}
        </button>

        <button
          style={micActive ? btnActive : btnBase}
          onClick={async () => {
            await toggleMic();
          }}
        >
          {micActive ? "Stop Mic" : "Start Mic"}
        </button>
      </div>
    </div>
  );
}
