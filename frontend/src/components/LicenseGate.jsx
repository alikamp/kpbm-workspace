import { useState } from 'react'
import { supabase } from '../lib/supabase'

const FEATURES = [
  {
    label: '200x Compute Acceleration',
    desc: 'Breaks the hardware barrier. KPBM delivers a 200x acceleration in time-to-solution compared to traditional solvers by maintaining absolute stability on highly optimized, lightweight grids.',
  },
  {
    label: 'Advanced Boundary Control',
    desc: 'Completely eliminates numerical divergence and simulation blowups at high velocities (Re ≥ 1500) using a proprietary, non-linear boundary stabilization layer.',
  },
  {
    label: 'Zero Infrastructure Overhead',
    desc: 'Runs high-fidelity, transient wake simulations directly on standard cloud infrastructure, completely bypassing the need for expensive multi-GPU setups or dedicated HPC clusters.',
  },
  {
    label: 'Uncompromised Physical Accuracy',
    desc: 'Captures precise, time-dependent structural forces, drag/lift histories, and authentic Strouhal frequency modes (St ≈ 0.20) without over-dampening the fluid physics.',
  },
]

export default function LicenseGate({ onVerified }) {
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleVerify = async () => {
    if (!code.trim()) return
    setLoading(true)
    setError('')

    try {
      const { data, error: dbError } = await supabase
        .from('licenses')
        .select('*')
        .eq('license_code', code.trim())
        .single()

      if (dbError || !data) {
        setError('License token not found.')
        setLoading(false)
        return
      }

      const now = new Date()
      const expiresAt = new Date(data.expires_at)

      if (data.status !== 'ACTIVE' || now > expiresAt) {
        setError('License token is expired or inactive.')
        setLoading(false)
        return
      }

      localStorage.setItem('kpbm_license', code.trim())
      localStorage.setItem('kpbm_license_tier', data.tier)
      onVerified(code.trim(), data.tier)
    } catch {
      setError('Verification failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-surface flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
            <span className="text-xs text-accent font-mono tracking-widest uppercase">KPBM Engine v1.0</span>
          </div>
          <h1 className="text-2xl font-bold text-white mb-2 tracking-tight">
            Simulation Console
          </h1>
          <p className="text-sm text-gray-500">
            Kinetic-Pressure Ballooning Model — 3D LBM Solver
          </p>
        </div>

        {/* Features */}
        <div className="mb-6 space-y-3">
          {FEATURES.map(f => (
            <div key={f.label} className="bg-panel border border-border rounded-lg px-5 py-4">
              <p className="text-xs text-accent font-bold uppercase tracking-wider mb-1">{f.label}</p>
              <p className="text-xs text-gray-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>

        {/* License input card */}
        <div className="bg-panel border border-border rounded-xl p-8">
          <p className="text-xs text-gray-400 mb-6 leading-relaxed">
            Enter your access license token to initialize the execution environment.
          </p>

          <div className="space-y-3 mb-6">
            <input
              type="text"
              value={code}
              onChange={e => setCode(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleVerify()}
              placeholder="KPBM-XXXX-XXXX-XXXX"
              className="w-full bg-surface border border-border rounded-lg px-4 py-3 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent transition-colors font-mono tracking-wider"
            />
            {error && (
              <p className="text-xs text-danger font-mono">{error}</p>
            )}
          </div>

          <button
            onClick={handleVerify}
            disabled={loading || !code.trim()}
            className="w-full bg-accent text-surface font-bold py-3 rounded-lg text-sm tracking-wide hover:bg-opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
          >
            {loading ? 'Verifying...' : 'Verify & Enter Console'}
          </button>
        </div>

        {/* Pricing info */}
        <div className="mt-6 border border-border rounded-xl p-6 bg-panel space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-white font-bold uppercase tracking-wider">Production License</p>
              <p className="text-xs text-gray-500 mt-0.5">30-day access · unlimited simulations</p>
            </div>
            <span className="text-lg font-bold text-accent font-mono">$7,995<span className="text-xs text-gray-500">/mo</span></span>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-white font-bold uppercase tracking-wider">Trial Token</p>
              <p className="text-xs text-gray-500 mt-0.5">24-hour single-use evaluation</p>
            </div>
            <span className="text-sm font-bold text-warn font-mono">On Request</span>
          </div>
          <div className="border-t border-border pt-4 space-y-2">
            <p className="text-xs text-gray-500 leading-relaxed">
              Payment via{' '}
              <span className="text-white font-semibold">Venmo @alika-p</span>
              {' '}or managed verification link. Licenses provisioned within 1 business day of transaction clearance.
            </p>
            <p className="text-xs text-gray-500">
              Trial requests:{' '}
              <a href="mailto:alikamp@gmail.com" className="text-accent hover:underline">alikamp@gmail.com</a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
