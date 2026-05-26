const GRID_PRESETS = [
  { label: '40 × 40 × 100  (Default)', Lx: 100, Ly: 40, Lz: 40 },
  { label: '50 × 50 × 120  (Medium)',  Lx: 120, Ly: 50, Lz: 50 },
  { label: '60 × 60 × 150  (High)',    Lx: 150, Ly: 60, Lz: 60 },
]

const STEP_OPTIONS = [500, 1000, 1500, 2000]

export default function ConfigPanel({ config, onChange, onExecute, disabled }) {
  const gridIndex = GRID_PRESETS.findIndex(
    g => g.Lx === config.Lx && g.Ly === config.Ly && g.Lz === config.Lz
  )

  const handleGridChange = e => {
    const preset = GRID_PRESETS[parseInt(e.target.value)]
    onChange({ ...config, Lx: preset.Lx, Ly: preset.Ly, Lz: preset.Lz })
  }

  return (
    <div className="bg-panel border border-border rounded-xl p-6 space-y-6">
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
        <h2 className="text-xs font-bold tracking-widest uppercase text-gray-400">
          Execution Parameters
        </h2>
      </div>

      {/* kpbm_alpha */}
      <div>
        <div className="flex justify-between mb-2">
          <label className="text-xs text-gray-400 uppercase tracking-wider">
            KPBM Alpha (α)
          </label>
          <span className="text-xs text-accent font-mono font-bold">
            {config.kpbm_alpha.toFixed(2)}
          </span>
        </div>
        <input
          type="range" min="0" max="1" step="0.01"
          value={config.kpbm_alpha}
          disabled={disabled}
          onChange={e => onChange({ ...config, kpbm_alpha: parseFloat(e.target.value) })}
          className="w-full accent-accent disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>0.00</span><span>1.00</span>
        </div>
      </div>

      {/* Re_target */}
      <div>
        <div className="flex justify-between mb-2">
          <label className="text-xs text-gray-400 uppercase tracking-wider">
            Reynolds Number (Re)
          </label>
          <span className="text-xs text-accent font-mono font-bold">
            {Math.round(config.Re_target)}
          </span>
        </div>
        <input
          type="range" min="500" max="3000" step="50"
          value={config.Re_target}
          disabled={disabled}
          onChange={e => onChange({ ...config, Re_target: parseFloat(e.target.value) })}
          className="w-full accent-accent disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>500</span><span>3000</span>
        </div>
      </div>

      {/* use_sponge */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-wider">Sponge Layer</p>
          <p className="text-xs text-gray-600 mt-0.5">Boundary absorption damping</p>
        </div>
        <button
          disabled={disabled}
          onClick={() => onChange({ ...config, use_sponge: !config.use_sponge })}
          className={`relative w-12 h-6 rounded-full transition-colors disabled:opacity-40 ${
            config.use_sponge ? 'bg-accent' : 'bg-border'
          }`}
        >
          <span
            className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
              config.use_sponge ? 'translate-x-7' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Grid Preset */}
      <div>
        <label className="text-xs text-gray-400 uppercase tracking-wider block mb-2">
          Grid Dimensions
        </label>
        <select
          value={gridIndex >= 0 ? gridIndex : 0}
          disabled={disabled}
          onChange={handleGridChange}
          className="w-full bg-surface border border-border rounded-lg px-3 py-2.5 text-sm text-white focus:outline-none focus:border-accent disabled:opacity-40 font-mono"
        >
          {GRID_PRESETS.map((g, i) => (
            <option key={i} value={i}>{g.label}</option>
          ))}
        </select>
      </div>

      {/* N_steps */}
      <div>
        <label className="text-xs text-gray-400 uppercase tracking-wider block mb-2">
          Timesteps (N)
        </label>
        <div className="grid grid-cols-4 gap-2">
          {STEP_OPTIONS.map(n => (
            <button
              key={n}
              disabled={disabled}
              onClick={() => onChange({ ...config, N_steps: n })}
              className={`py-2 rounded-lg text-xs font-mono font-bold border transition-all disabled:opacity-40 ${
                config.N_steps === n
                  ? 'border-accent text-accent bg-accent bg-opacity-10'
                  : 'border-border text-gray-500 hover:border-gray-500'
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Execute */}
      <button
        onClick={onExecute}
        disabled={disabled}
        className="w-full bg-accent text-surface font-bold py-3.5 rounded-lg text-sm tracking-widest uppercase hover:bg-opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
      >
        {disabled ? 'Simulation Running...' : '▶  Execute Simulation'}
      </button>
    </div>
  )
}
