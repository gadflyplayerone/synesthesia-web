import { useEffect, useState } from 'react'
import Visualizer from './components/Visualizer'
import { useMicAnalyser } from './audio/useMicAnalyser'

export default function App() {
  const [autostart, setAutostart] = useState(true)
  const [smoothing, setSmoothing] = useState(0.85)
  const [sensitivity, setSensitivity] = useState(1.0)
  const [running, setRunning] = useState(false)
  const [uiHidden, setUiHidden] = useState(true)

  const analyser = useMicAnalyser({
    fftSize: 1024,
    smoothingTimeConstant: 0.8
  })

  useEffect(() => {
    if (autostart && !running) {
      analyser.start().then(ok => setRunning(ok)).catch(() => setRunning(false))
      setAutostart(false)
    }
  }, [autostart])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'h') setUiHidden(h => !h)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const handleStart = async () => {
    const ok = await analyser.start()
    setRunning(ok)
  }
  const handleStop = () => {
    analyser.stop()
    setRunning(false)
  }

  return (
    <div className="wrap">
      {!uiHidden && (
        <div className="topbar">
          <div className="controls">
            <label>
              Smoothing
              <input type="range" min="0" max="0.98" step="0.01"
                value={smoothing}
                onChange={e => setSmoothing(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Sensitivity
              <input type="range" min="0.2" max="3.0" step="0.05"
                value={sensitivity}
                onChange={e => setSensitivity(parseFloat(e.target.value))}
              />
            </label>
          </div>
          <div className="actions">
            {!running ? (
              <button onClick={handleStart}>Enable Microphone</button>
            ) : (
              <>
                <button className="secondary" onClick={handleStop}>Stop</button>
              </>
            )}
            <button onClick={() => setUiHidden(true)}>Hide UI</button>
          </div>
        </div>
      )}

      {uiHidden && (
        <button className="floating-toggle" onClick={() => setUiHidden(false)}>
          Show UI (H)
        </button>
      )}

      <Visualizer
        className="canvas"
        analyser={analyser}
        smoothing={smoothing}
        sensitivity={sensitivity}
      />

      <div className="hint">
        {running
          ? "Speak, sing, or play music near the mic · Best in Chrome/Edge/Firefox · Press H to toggle UI"
          : "Click 'Enable Microphone' to begin · Your audio never leaves the browser · Press H to toggle UI"}
      </div>
    </div>
  )
}