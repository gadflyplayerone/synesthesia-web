
// src/audio/pitch.ts
export type PitchResult = {
  frequency: number | null;
  midi: number | null;
  note: string | null;
  clarity: number; // 0..1 confidence
};

const NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];

export function freqToMidi(f: number): number {
  return 69 + 12 * Math.log2(f / 440);
}
export function midiToFreq(m: number): number {
  return 440 * Math.pow(2, (m-69)/12);
}
export function midiToNoteName(m: number): string {
  const n = Math.round(m);
  const name = NOTE_NAMES[(n % 12 + 12) % 12];
  const octave = Math.floor(n/12) - 1;
  return `${name}${octave}`;
}

// Autocorrelation pitch detection on time-domain signal (Uint8Array or Float32Array)
export function detectPitchACF(time: Float32Array, sampleRate: number, minFreq=50, maxFreq=1000): PitchResult {
  // remove DC, normalize
  const buf = new Float32Array(time.length);
  let mean = 0;
  for (let i=0;i<time.length;i++) mean += time[i];
  mean /= time.length;
  let maxAbs = 1e-9;
  for (let i=0;i<time.length;i++) { const v = time[i]-mean; buf[i]=v; maxAbs=Math.max(maxAbs, Math.abs(v)); }
  for (let i=0;i<buf.length;i++) buf[i]/=maxAbs;

  const minLag = Math.floor(sampleRate/maxFreq);
  const maxLag = Math.floor(sampleRate/minFreq);
  let bestLag = -1, bestVal = 0;

  for (let lag=minLag; lag<=maxLag; lag++) {
    let sum = 0;
    for (let i=0;i<buf.length-lag;i++) {
      sum += buf[i]*buf[i+lag];
    }
    const val = sum / (buf.length - lag);
    if (val > bestVal) { bestVal = val; bestLag = lag; }
  }

  if (bestLag <= 0) return { frequency: null, midi: null, note: null, clarity: 0 };

  const freq = sampleRate / bestLag;
  const midi = freqToMidi(freq);
  const note = midiToNoteName(midi);
  const clarity = Math.min(1, Math.max(0, bestVal));

  return { frequency: freq, midi, note, clarity };
}

// Cepstrum-like "second-order FFT" taste: FFT magnitude -> log -> IFFT to quefrency,
// find peak around voice region to estimate pitch; lightweight DFT/IFFT (N small)
export function simpleCepstrumPitch(mag: Float32Array, sampleRate: number): PitchResult {
  const N = mag.length;
  const logMag = new Float32Array(N);
  for (let i=0;i<N;i++) logMag[i] = Math.log(1e-12 + Math.max(0, mag[i]));
  // naive DFT (only cosine for real cepstrum simplified)
  const cep = new Float32Array(N);
  for (let k=0;k<N;k++) {
    let sum = 0;
    for (let n=0;n<N;n++) sum += logMag[n]*Math.cos((2*Math.PI*k*n)/N);
    cep[k] = sum;
  }
  // search for peak within plausible quefrency window (2ms..20ms => 50..500Hz)
  const qMin = Math.floor(sampleRate/500); // period samples ~ 2ms
  const qMax = Math.min(N-1, Math.floor(sampleRate/50)); // ~20ms
  let bestK = -1, bestV = -1e9;
  for (let k=qMin;k<=qMax;k++){
    if (cep[k] > bestV) { bestV = cep[k]; bestK = k; }
  }
  if (bestK < 1) return { frequency: null, midi: null, note: null, clarity: 0 };
  const freq = sampleRate / bestK;
  const midi = freqToMidi(freq);
  const note = midiToNoteName(midi);
  const clarity = Math.max(0, Math.min(1, (bestV - 0) / (1000 + Math.abs(bestV))));
  return { frequency: freq, midi, note, clarity };
}

// Lightweight predictive model: given last N MIDI notes, predict next by trend
// If clarity too low we skip. Uses median of intervals to reduce jitter.
export function predictNextMidi(history: number[], horizon=1): number | null {
  const xs = history.slice(-8); // last up to 8 notes
  if (xs.length < 3) return null;
  const intervals = [];
  for (let i=1;i<xs.length;i++) intervals.push(xs[i]-xs[i-1]);
  intervals.sort((a,b)=>a-b);
  const median = intervals[Math.floor(intervals.length/2)] || 0;
  const last = xs[xs.length-1];
  return last + median * horizon;
}
