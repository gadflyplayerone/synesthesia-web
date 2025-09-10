import { useEffect, useRef } from 'react'
import type { MicAnalyser } from '../audio/useMicAnalyser'

type Props = {
  analyser: MicAnalyser
  className?: string
  smoothing?: number
  sensitivity?: number
}

function dbToUnit(db: number): number {
  const min = -100, max = -10
  const clamped = Math.max(min, Math.min(max, db))
  return (clamped - min) / (max - min)
}
const DEG = Math.PI / 180
const clamp01 = (x: number) => x < 0 ? 0 : (x > 1 ? 1 : x)
function hash01(x: number, t: number) { const s = Math.sin((x * 12.9898 + t * 0.35) * 43758.5453); return s - Math.floor(s) }

export default function Visualizer({ analyser, className, smoothing = 0.85, sensitivity = 1.0 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const emaSpectrumRef = useRef<Float32Array | null>(null)
  const timeBufRef = useRef<Uint8Array | null>(null)
  const freqBufRef = useRef<Float32Array | null>(null)
  const tRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d', { alpha: false })!
    const dpr = Math.max(1, window.devicePixelRatio || 1)

    function resize() {
      const { clientWidth, clientHeight } = canvas
      canvas.width = Math.floor(clientWidth * dpr)
      canvas.height = Math.floor(clientHeight * dpr)
    }
    const onResize = () => resize()
    const onVis = () => { if (document.visibilityState === 'hidden') cancel(); else loop() }
    resize()
    window.addEventListener('resize', onResize)
    document.addEventListener('visibilitychange', onVis)

    let last = performance.now()
    const loop = () => {
      const node = analyser.node()
      const W = canvas.width, H = canvas.height
      const now = performance.now()
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      tRef.current += dt

      if (!node) {
        ctx.fillStyle = '#0a0e2a'; ctx.fillRect(0,0,W,H)
        rafRef.current = requestAnimationFrame(loop); return
      }

      const bins = analyser.frequencyBinCount()
      if (!freqBufRef.current || freqBufRef.current.length !== bins) {
        freqBufRef.current = new Float32Array(bins)
        emaSpectrumRef.current = new Float32Array(bins)
      }
      if (!timeBufRef.current || timeBufRef.current.length !== node.fftSize) {
        timeBufRef.current = new Uint8Array(node.fftSize)
      }

      analyser.getFloatFrequency(freqBufRef.current!)
      analyser.getTimeDomain(timeBufRef.current!)

      const freq = freqBufRef.current!
      const ema = emaSpectrumRef.current!
      for (let i=0;i<freq.length;i++){ const u = dbToUnit(freq[i]) * sensitivity; ema[i] = ema[i]*smoothing + u*(1-smoothing) }

      draw(ctx, W, H, ema, timeBufRef.current!, node.fftSize, tRef.current, analyser.sampleRate() ?? 48000)
      rafRef.current = requestAnimationFrame(loop)
    }

    const cancel = () => { if (rafRef.current !== null) cancelAnimationFrame(rafRef.current); rafRef.current = null }
    loop()
    return () => { cancel(); window.removeEventListener('resize', onResize); document.removeEventListener('visibilitychange', onVis) }
  }, [analyser, smoothing, sensitivity])

  return <canvas ref={canvasRef} className={className} />
}

function draw(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  spectrum: Float32Array,
  time: Uint8Array,
  fftSize: number,
  t: number,
  sampleRate: number
){
  ctx.fillStyle = '#0a0e2a'; ctx.fillRect(0,0,W,H)
  const cx = W*0.5, cy = H*0.5
  const orbR = Math.min(W,H)*0.35

  // RMS loudness
  let rms = 0; for(let i=0;i<time.length;i++){ const x = (time[i]-128)/128; rms += x*x } rms = Math.sqrt(rms/time.length)

  // Outer halo
  const baseR_out = orbR*1.1, maxR = Math.min(W,H)*0.52
  const n = spectrum.length, step = Math.max(1, Math.floor(n/240))
  ctx.save(); ctx.translate(cx,cy)
  for(let i=0;i<n;i+=step){
    const pct=i/n, mag=Math.pow(spectrum[i],1.35)
    const r = baseR_out + mag*(maxR-baseR_out)
    const th = pct*Math.PI*2
    const x1=Math.cos(th)*baseR_out, y1=Math.sin(th)*baseR_out
    const x2=Math.cos(th)*r,         y2=Math.sin(th)*r
    const hue = 210 + pct*120
    ctx.strokeStyle = `hsl(${hue},70%,${Math.min(78,35+mag*55)}%)`
    ctx.lineWidth = Math.max(1, mag*5.0)
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke()
  }
  ctx.restore()

  // Orb
  const glow = ctx.createRadialGradient(cx,cy,0,cx,cy,orbR*1.35)
  glow.addColorStop(0,'rgba(255,216,77,0.22)'); glow.addColorStop(1,'rgba(255,216,77,0.00)')
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(cx,cy,orbR*1.35,0,Math.PI*2); ctx.fill()
  ctx.fillStyle = 'rgba(5,8,26,0.92)'; ctx.beginPath(); ctx.arc(cx,cy,orbR,0,Math.PI*2); ctx.fill()
// In-orb FFT (white upward + slanted bluish downward)
// ── now the slanted tails pivot from the horizontal centerline (y = axisY)
//    of each vertical bar and slant to the bottom-left.
ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, orbR * 0.97, 0, Math.PI * 2); ctx.clip();

const bars  = Math.min(160, spectrum.length);
const stride = Math.max(1, Math.floor(spectrum.length / bars));
const halfW = orbR * 0.92;
const left  = cx - halfW;
const width = 2 * halfW;
const barW  = width / bars;
const axisY = cy;

const SCALE = 1.25;
const tilt  = -18 * DEG; // negative => slant toward bottom-left

ctx.globalCompositeOperation = 'lighter';
for (let bi = 0, i = 0; bi < bars; bi++, i += stride) {
  const u     = clamp01(spectrum[i] * SCALE);
  const alpha = 0.18 + 0.55 * u + 0.25 * hash01(bi, t);

  const x = left + bi * barW;
  const w = Math.max(2, barW * 0.8);

  // Upward (white) bar from horizontal axis
  const hUp = u * orbR * 0.9;
  ctx.fillStyle = `rgba(255,255,255,${alpha})`;
  ctx.fillRect(x, axisY - hUp, w, hUp);

  // Slanted (bluish) tail: pivot at the bar's midpoint on the horizontal axis
  const hDown   = u * orbR * 0.25;
  const anchorX = x + w;

  ctx.save();
  ctx.translate(anchorX, axisY); // pivot at current bar center on y = axisY
  ctx.rotate(tilt);              // rotate local coords so +y points down-left
  ctx.fillStyle = `rgba(170,188,255,${alpha * 0.7})`;
  ctx.fillRect(-Math.max(2, barW * 0.75) / 2, 0, Math.max(2, barW * 0.75), hDown);
  ctx.restore();
}
ctx.globalCompositeOperation = 'source-over';
ctx.restore();

ctx.strokeStyle = 'rgba(255,216,77,0.55)';
ctx.lineWidth = 2;
ctx.beginPath();
ctx.arc(cx, cy, orbR, 0, Math.PI * 2);
ctx.stroke();

  // Yellow sinusoid only inside orb
  const hzNy = sampleRate/2, binHz = hzNy/spectrum.length
  function band(h0:number,h1:number){ const i0=Math.max(0,Math.floor(h0/binHz)), i1=Math.min(spectrum.length-1,Math.ceil(h1/binHz)); let s=0; for(let i=i0;i<=i1;i++) s+=spectrum[i]; return s/Math.max(1,i1-i0+1) }
  const A1 = band(80,300), A2 = band(300,1200), A3 = band(1200,3000), A4 = band(3000,6000)
  const amp = (0.30 + 1.05*rms)*orbR*0.95
  const speed = 0.45, kx = 2*Math.PI*(0.55 + 0.9*A2)
  ctx.save(); ctx.beginPath(); ctx.arc(cx,cy,orbR*0.98,0,Math.PI*2); ctx.clip()
  ctx.lineJoin='round'; ctx.lineCap='round'
  const R = cx-orbR, steps = Math.floor(2*orbR)
  ctx.strokeStyle='rgba(255,216,77,0.95)'; ctx.lineWidth=Math.max(2.2,2.6+2.4*(A1+A2))
  ctx.beginPath()
  for(let s=0;s<=steps;s++){ const x=R+(s/steps)*(2*orbR); const ph=2*Math.PI*speed*t + (x-R)*kx/(2*orbR)
    const y0 = (A1*Math.sin(ph)) + (0.8*A2*Math.sin(2*ph+0.25)) + (0.45*A3*Math.sin(3*ph+0.9)) + (0.25*A4*Math.sin(4*ph+1.6))
    const y = cy + y0*amp; if(s===0) ctx.moveTo(x,y); else ctx.lineTo(x,y) }
  ctx.stroke()
  ctx.strokeStyle='rgba(255,216,77,0.3)'
  for(let k=1;k<=10;k++){ ctx.beginPath()
    for(let s=0;s<=steps;s++){ const x=R+(s/steps)*(2*orbR); const ph=2*Math.PI*speed*(t - k*0.008) + (x-R)*kx/(2*orbR)
      const y0 = (1.2*A1*Math.sin(ph)) + (0.9*A2*Math.sin(2*ph+0.2)) + (0.5*A3*Math.sin(3*ph+0.8))
      const y = cy + y0*(amp*(1.0-0.05*k)); if(s===0) ctx.moveTo(x,y); else ctx.lineTo(x,y) }
    ctx.stroke() }
  ctx.restore()
}