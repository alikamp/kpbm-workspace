import { useEffect, useRef } from 'react'

export default function ExecutionPanel({ job }) {
  const logRef = useRef(null)

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [job?.log_stream])

  const progress = job?.progress ?? 0
  const currentStep = job?.current_step ?? 0
  const log = job?.log_stream ?? 'Initializing...'

  return (
    <div className="bg-panel border border-border rounded-xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          <h2 className="text-xs font-bold tracking-widest uppercase text-gray-400">
            Execution Monitor
          </h2>
        </div>
        <span className="text-xs font-mono text-accent font-bold">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div>
        <div className="w-full bg-surface rounded-full h-2 border border-border overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-600 mt-1.5 font-mono">
          <span>Step {currentStep.toLocaleString()}</span>
          <span>{progress}% complete</span>
        </div>
      </div>

      {/* Log stream */}
      <div>
        <p className="text-xs text-gray-600 uppercase tracking-wider mb-2">Live Feed</p>
        <div
          ref={logRef}
          className="bg-surface border border-border rounded-lg p-4 h-48 overflow-y-auto font-mono text-xs text-gray-400 leading-relaxed"
        >
          {log.split('\n').map((line, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-gray-700 select-none shrink-0">›</span>
              <span>{line}</span>
            </div>
          ))}
          <div className="flex gap-2 mt-1">
            <span className="text-gray-700 select-none">›</span>
            <span className="text-accent animate-pulse">_</span>
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-surface border border-border rounded-lg p-3">
          <p className="text-xs text-gray-600 uppercase tracking-wider mb-1">Status</p>
          <p className="text-sm font-mono font-bold text-accent">RUNNING</p>
        </div>
        <div className="bg-surface border border-border rounded-lg p-3">
          <p className="text-xs text-gray-600 uppercase tracking-wider mb-1">Timestep</p>
          <p className="text-sm font-mono font-bold text-white">{currentStep.toLocaleString()}</p>
        </div>
      </div>
    </div>
  )
}
