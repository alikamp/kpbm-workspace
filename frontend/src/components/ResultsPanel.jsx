import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'

function buildChartData(drag, lift) {
  const len = Math.max(drag.length, lift.length)
  return Array.from({ length: len }, (_, i) => ({
    step: i + 1,
    drag: drag[i] ?? null,
    lift: lift[i] ?? null,
  }))
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-panel border border-border rounded-lg px-3 py-2 text-xs font-mono">
      <p className="text-gray-400 mb-1">Step {label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey}: {p.value?.toFixed(5)}
        </p>
      ))}
    </div>
  )
}

export default function ResultsPanel({ job, onReset }) {
  const stable = job.status === 'STABLE'
  const drag = job.drag_history ?? []
  const lift = job.lift_history ?? []
  const chartData = buildChartData(drag, lift)

  const avgDrag = drag.length ? (drag.reduce((a, b) => a + b, 0) / drag.length).toFixed(5) : '—'
  const avgLift = lift.length ? (lift.reduce((a, b) => a + b, 0) / lift.length).toFixed(5) : '—'

  return (
    <div className="space-y-5">
      {/* Status header */}
      <div className={`border rounded-xl p-5 flex items-center justify-between ${
        stable
          ? 'border-accent bg-accent bg-opacity-5'
          : 'border-danger bg-danger bg-opacity-5'
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${stable ? 'bg-accent' : 'bg-danger'}`} />
          <div>
            <p className={`text-lg font-bold font-mono tracking-widest ${stable ? 'text-accent' : 'text-danger'}`}>
              {stable ? 'STABLE' : 'CRASHED'}
            </p>
            <p className="text-xs text-gray-500 mt-0.5">
              {stable
                ? `Simulation completed at step ${job.current_step?.toLocaleString()}`
                : `Numerical divergence detected at step ${job.current_step?.toLocaleString()}`}
            </p>
          </div>
        </div>
        <button
          onClick={onReset}
          className="text-xs border border-border text-gray-400 px-4 py-2 rounded-lg hover:border-gray-500 hover:text-white transition-all font-mono"
        >
          New Run
        </button>
      </div>

      {/* Crash detail */}
      {!stable && (
        <div className="bg-panel border border-danger border-opacity-40 rounded-xl p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Divergence Log</p>
          <p className="text-sm font-mono text-danger">
            Simulation Unzipped Prematurely at Step {job.current_step?.toLocaleString()}
          </p>
          <p className="text-xs text-gray-600 mt-2 font-mono">{job.log_stream}</p>
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Final Step', value: job.current_step?.toLocaleString() ?? '—', color: 'text-white' },
          { label: 'Avg Drag Coeff', value: avgDrag, color: 'text-accent' },
          { label: 'Avg Lift Coeff', value: avgLift, color: 'text-warn' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-panel border border-border rounded-xl p-4">
            <p className="text-xs text-gray-600 uppercase tracking-wider mb-1">{label}</p>
            <p className={`text-sm font-mono font-bold ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="space-y-4">
          {/* Drag history */}
          <div className="bg-panel border border-border rounded-xl p-5">
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-4">
              Drag Force History
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#30363d" strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: '#6e7681', fontSize: 10, fontFamily: 'monospace' }}
                  tickLine={false}
                  label={{ value: 'Timestep', position: 'insideBottom', offset: -2, fill: '#6e7681', fontSize: 10 }}
                />
                <YAxis
                  tick={{ fill: '#6e7681', fontSize: 10, fontFamily: 'monospace' }}
                  tickLine={false}
                  width={55}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#30363d" />
                <Line
                  type="monotone" dataKey="drag"
                  stroke="#00d4aa" dot={false} strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Lift history */}
          <div className="bg-panel border border-border rounded-xl p-5">
            <p className="text-xs text-gray-400 uppercase tracking-wider mb-4">
              Lift Force History
            </p>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#30363d" strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: '#6e7681', fontSize: 10, fontFamily: 'monospace' }}
                  tickLine={false}
                  label={{ value: 'Timestep', position: 'insideBottom', offset: -2, fill: '#6e7681', fontSize: 10 }}
                />
                <YAxis
                  tick={{ fill: '#6e7681', fontSize: 10, fontFamily: 'monospace' }}
                  tickLine={false}
                  width={55}
                />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#30363d" />
                <Line
                  type="monotone" dataKey="lift"
                  stroke="#e3b341" dot={false} strokeWidth={1.5}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
