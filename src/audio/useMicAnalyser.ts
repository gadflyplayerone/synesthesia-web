import { useCallback, useRef, useState } from 'react'

export type MicAnalyser = {
  ready: boolean
  node: () => AnalyserNode | null
  start: () => Promise<boolean>
  stop: () => void
  getFloatFrequency: (arr: Float32Array) => void
  getTimeDomain: (arr: Uint8Array) => void
  sampleRate: () => number | null
  fftSize: number
  frequencyBinCount: () => number
}

type Options = {
  fftSize?: 256 | 512 | 1024 | 2048 | 4096
  smoothingTimeConstant?: number
}

export function useMicAnalyser(opts: Options = {}) {
  const fftSize = opts.fftSize ?? 1024
  const smoothingTimeConstant = opts.smoothingTimeConstant ?? 0.8

  const ctxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [ready, setReady] = useState(false)

  const getOrCreate = useCallback(async (): Promise<AnalyserNode> => {
    if (analyserRef.current && ctxRef.current && streamRef.current) return analyserRef.current
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1, sampleRate: { ideal: 48000 } },
      video: false
    })
    const Ctx = (window.AudioContext || (window as any).webkitAudioContext)
    const ctx: AudioContext = new Ctx()
    const src = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = fftSize
    analyser.smoothingTimeConstant = smoothingTimeConstant
    analyser.minDecibels = -100
    analyser.maxDecibels = -10
    src.connect(analyser)
    streamRef.current = stream
    ctxRef.current = ctx
    analyserRef.current = analyser
    setReady(true)
    return analyser
  }, [fftSize, smoothingTimeConstant])

  const start = useCallback(async (): Promise<boolean> => {
    try {
      await getOrCreate()
      await ctxRef.current!.resume()
      setReady(true)
      return true
    } catch {
      setReady(false)
      return false
    }
  }, [getOrCreate])

  const stop = useCallback(() => {
    try {
      if (streamRef.current) {
        for (const t of streamRef.current.getTracks()) t.stop()
        streamRef.current = null
      }
      if (ctxRef.current) ctxRef.current.suspend()
    } finally { setReady(false) }
  }, [])

// TS 5.4 typed arrays are generic (…<ArrayBufferLike>), while WebAudio expects …<ArrayBuffer>.
// Cast at the boundary; the underlying memory layout is identical.
const getFloatFrequency = useCallback((arr: Float32Array) => {
  const a = analyserRef.current; if (!a) return;
  a.getFloatFrequencyData(arr as unknown as Float32Array<ArrayBuffer>);
}, []);

const getTimeDomain = useCallback((arr: Uint8Array) => {
  const a = analyserRef.current; if (!a) return;
  a.getByteTimeDomainData(arr as unknown as Uint8Array<ArrayBuffer>);
}, []);


  const node = useCallback(() => analyserRef.current, [])
  const sampleRate = useCallback(() => ctxRef.current?.sampleRate ?? null, [])
  const frequencyBinCount = useCallback(() => analyserRef.current?.frequencyBinCount ?? (fftSize / 2), [fftSize])

  return { ready, node, start, stop, getFloatFrequency, getTimeDomain, sampleRate, fftSize, frequencyBinCount }
}