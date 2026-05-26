import { useState } from 'react'
import { supabase } from '../lib/supabase'

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
    <div className="min-h-screen bg-surface flex items-center justify-center px-4">
      <div className="w-full max-w-md">
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

        {/* Card */}
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
        <div className="mt-6 border border-border rounded-xl p-6 bg-panel">
          <p className="text-xs text-gray-500 leading-relaxed">
            To request a{' '}
            <span className="text-warn">24-hour single-use evaluation token</span>, or to extend a
            production license for 30 days{' '}
            <span className="text-white font-semibold">($7,995/mo)</span>, please contact
            engineering administration. Payments accepted via managed verification link or
            corporate Venmo transfer. Licenses are provisioned within 1 business day of
            transaction clearance.
          </p>
        </div>
      </div>
    </div>
  )
}
