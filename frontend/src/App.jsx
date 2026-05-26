import { useState, useEffect, useRef, useCallback } from 'react'
import { supabase } from './lib/supabase'
import LicenseGate from './components/LicenseGate'
import ConfigPanel from './components/ConfigPanel'
import ExecutionPanel from './components/ExecutionPanel'
import ResultsPanel from './components/ResultsPanel'

const API_URL = import.meta.env.VITE_API_URL

const DEFAULT_CONFIG = {
  kpbm_alpha: 0.25,
  use_sponge: true,
  Re_target: 1500,
  Lx: 100, Ly: 40, Lz: 40,
  U_inf: 0.04,
  N_steps: 1500,
}

// STATE: 'GATE' | 'CONFIG' | 'RUNNING' | 'COMPLETE'

export default function App() {
  const [appState, setAppState] = useState('GATE')
  const [license, setLicense] = useState(null)
  const [licenseTier, setLicenseTier] = useState(null)
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [jobId, setJobId] = useState(null)
  const [jobData, setJobData] = useState(null)
  const [submitError, setSubmitError] = useState('')
  const pollRef = useRef(null)

  // Restore session from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('kpbm_license')
    const tier = localStorage.getItem('kpbm_license_tier')
    if (saved) {
      setLicense(saved)
      setLicenseTier(tier)
      setAppState('CONFIG')
    }
  }, [])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback((id) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      const { data, error } = await supabase
        .from('jobs')
        .select('*')
        .eq('id', id)
        .single()

      if (error || !data) return

      setJobData(data)

      if (data.status === 'STABLE' || data.status === 'CRASHED') {
        stopPolling()
        setAppState('COMPLETE')
      }
    }, 2000)
  }, [stopPolling])

  useEffect(() => () => stopPolling(), [stopPolling])

  const handleVerified = (code, tier) => {
    setLicense(code)
    setLicenseTier(tier)
    setAppState('CONFIG')
  }

  const handleExecute = async () => {
    setSubmitError('')
    setAppState('RUNNING')
    setJobData(null)

    try {
      const res = await fetch(`${API_URL}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...config, license_code: license }),
      })

      if (res.status === 403) {
        setSubmitError('License rejected by server. Token may be expired.')
        setAppState('CONFIG')
        return
      }

      if (!res.ok) {
        setSubmitError(`Server error: ${res.status}`)
        setAppState('CONFIG')
        return
      }

      const { job_id } = await res.json()
      setJobId(job_id)
      startPolling(job_id)
    } catch (err) {
      setSubmitError(`Connection failed: ${err.message}`)
      setAppState('CONFIG')
    }
  }

  const handleReset = () => {
    stopPolling()
    setJobId(null)
    setJobData(null)
    setSubmitError('')
    setAppState('CONFIG')
  }

  const handleLogout = () => {
    localStorage.removeItem('kpbm_license')
    localStorage.removeItem('kpbm_license_tier')
    stopPolling()
    setLicense(null)
    setJobId(null)
    setJobData(null)
    setAppState('GATE')
  }

  if (appState === 'GATE') {
    return <LicenseGate onVerified={handleVerified} />
  }

  return (
    <div className="min-h-screen bg-surface">
      {/* Top bar */}
      <header className="border-b border-border px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent" />
          <span className="text-sm font-bold tracking-widest uppercase text-white">
            KPBM Engine
          </span>
          <span className="text-xs text-gray-600">/ Simulation Console</span>
        </div>
        <div className="flex items-center gap-4">
          <span className={`text-xs font-mono px-2.5 py-1 rounded-full border ${
            licenseTier === 'PAID'
              ? 'border-accent text-accent bg-accent bg-opacity-10'
              : 'border-warn text-warn bg-warn bg-opacity-10'
          }`}>
            {licenseTier === 'PAID' ? 'PRODUCTION' : 'TRIAL'}
          </span>
          <button
            onClick={handleLogout}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors font-mono"
          >
            Sign Out
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {/* Page title */}
        <div className="mb-8">
          <h1 className="text-xl font-bold text-white tracking-tight">
            3D Lattice Boltzmann Simulation
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Kinetic-Pressure Ballooning Model — Non-linear stabilization for Re ≥ 1500
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: config always visible */}
          <div className="lg:col-span-1">
            <ConfigPanel
              config={config}
              onChange={setConfig}
              onExecute={handleExecute}
              disabled={appState === 'RUNNING'}
            />
            {submitError && (
              <p className="mt-3 text-xs text-danger font-mono bg-panel border border-danger border-opacity-30 rounded-lg px-4 py-3">
                {submitError}
              </p>
            )}
          </div>

          {/* Right: state-driven content */}
          <div className="lg:col-span-2">
            {appState === 'CONFIG' && (
              <div className="bg-panel border border-border rounded-xl p-10 flex flex-col items-center justify-center text-center h-full min-h-64">
                <div className="w-12 h-12 rounded-full border border-border flex items-center justify-center mb-4">
                  <span className="text-accent text-xl">▷</span>
                </div>
                <p className="text-sm text-gray-400 font-mono">
                  Configure parameters and execute simulation
                </p>
                <p className="text-xs text-gray-600 mt-2">
                  Simulation runs asynchronously — ~75–90s for 1500 steps
                </p>
              </div>
            )}

            {appState === 'RUNNING' && (
              <ExecutionPanel job={jobData} />
            )}

            {appState === 'COMPLETE' && jobData && (
              <ResultsPanel job={jobData} onReset={handleReset} />
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
