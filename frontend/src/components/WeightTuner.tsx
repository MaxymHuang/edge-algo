import { useState, useEffect, useRef } from 'react'
import CameraFeed from './CameraFeed'
import EdgeDetection from './EdgeDetection'

interface TunerStatus {
  type: string
  human_angle: number
  predicted_angle: number
  sample_count: number
  is_recording: boolean
  densities: {
    far_left: number
    left: number
    center: number
    right: number
    far_right: number
  }
  optimization_result: {
    weights: number[]
    mae: number
    outer: number
    inner: number
    center: number
  } | null
}

const STEERING_ANGLES = [-45, -30, -15, 0, 15, 30, 45]

const WeightTuner = () => {
  const [status, setStatus] = useState<TunerStatus>({
    type: 'status',
    human_angle: 0,
    predicted_angle: 0,
    sample_count: 0,
    is_recording: false,
    densities: {
      far_left: 0,
      left: 0,
      center: 0,
      right: 0,
      far_right: 0
    },
    optimization_result: null
  })
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/api/tuner`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'status') {
          setStatus(data)
        }
      } catch (error) {
        console.error('Error parsing tuner status:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onopen = () => {
      console.log('Tuner WebSocket connected')
    }

    return () => {
      ws.close()
    }
  }, [])

  const setAngle = async (angle: number) => {
    try {
      const response = await fetch('/api/set_angle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ angle })
      })
      const result = await response.json()
      if (!response.ok) {
        alert(`Failed to set angle: ${result.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Error setting angle:', error)
      alert('Failed to set angle')
    }
  }

  const startRecording = async () => {
    try {
      const response = await fetch('/api/start_recording', {
        method: 'POST'
      })
      if (!response.ok) {
        const error = await response.json()
        alert(`Failed to start recording: ${error.detail}`)
      }
    } catch (error) {
      console.error('Error starting recording:', error)
      alert('Failed to start recording')
    }
  }

  const stopRecording = async () => {
    try {
      const response = await fetch('/api/stop_recording', {
        method: 'POST'
      })
      if (!response.ok) {
        const error = await response.json()
        alert(`Failed to stop recording: ${error.detail}`)
      }
    } catch (error) {
      console.error('Error stopping recording:', error)
      alert('Failed to stop recording')
    }
  }

  const runOptimization = async () => {
    if (status.sample_count < 10) {
      alert(`Insufficient samples: ${status.sample_count}. Need at least 10 samples.`)
      return
    }

    setIsOptimizing(true)
    try {
      const response = await fetch('/api/run_optimization', {
        method: 'POST'
      })
      const result = await response.json()
      if (!response.ok) {
        alert(`Optimization failed: ${result.detail}`)
      } else {
        alert(`Optimization complete! Best MAE: ${result.result.mae.toFixed(2)}°`)
      }
    } catch (error) {
      console.error('Error running optimization:', error)
      alert('Failed to run optimization')
    } finally {
      setIsOptimizing(false)
    }
  }

  const saveWeights = async () => {
    if (!status.optimization_result) {
      alert('No optimization result to save. Run optimization first.')
      return
    }

    setIsSaving(true)
    try {
      const response = await fetch('/api/save_weights', {
        method: 'POST'
      })
      const result = await response.json()
      if (!response.ok) {
        alert(`Failed to save: ${result.detail}`)
      } else {
        alert('Weights saved successfully to config.py!')
      }
    } catch (error) {
      console.error('Error saving weights:', error)
      alert('Failed to save weights')
    } finally {
      setIsSaving(false)
    }
  }

  const resetSamples = async () => {
    if (!confirm('Are you sure you want to clear all collected samples?')) {
      return
    }

    try {
      const response = await fetch('/api/reset_samples', {
        method: 'POST'
      })
      if (!response.ok) {
        const error = await response.json()
        alert(`Failed to reset: ${error.detail}`)
      }
    } catch (error) {
      console.error('Error resetting samples:', error)
      alert('Failed to reset samples')
    }
  }

  const formatPercentage = (value: number): string => {
    return (value * 100).toFixed(2) + '%'
  }

  const getAngleColor = (angle: number) => {
    if (angle < 0) return 'bg-blue-500 hover:bg-blue-600'
    if (angle > 0) return 'bg-red-500 hover:bg-red-600'
    return 'bg-green-500 hover:bg-green-600'
  }

  const getAngleText = (angle: number) => {
    return angle > 0 ? `+${angle}°` : `${angle}°`
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-center">Weight Tuner - Bayesian Optimization</h1>

        {/* Status Panel */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">Current Status</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">Human Angle</div>
              <div className="text-2xl font-bold">{getAngleText(status.human_angle)}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">Predicted Angle</div>
              <div className="text-2xl font-bold">{getAngleText(status.predicted_angle)}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">Samples Collected</div>
              <div className="text-2xl font-bold">{status.sample_count}</div>
            </div>
          </div>

          {/* Density Visualization */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">Region Densities</h3>
            <div className="flex items-center gap-2">
              {['far_left', 'left', 'center', 'right', 'far_right'].map((region) => {
                const density = status.densities[region as keyof typeof status.densities]
                const chars = "▁▂▃▄▅▆▇█"
                const charIdx = Math.min(Math.floor(density * chars.length * 10), chars.length - 1)
                return (
                  <div key={region} className="flex-1 text-center">
                    <div className="text-xs text-gray-400 mb-1">{region.replace('_', '-')}</div>
                    <div className="text-2xl mb-1">{chars[charIdx]}</div>
                    <div className="text-xs font-mono">{formatPercentage(density)}</div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Steering Controls */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">Steering Controls</h2>
          
          <div className="mb-4">
            <div className="text-sm text-gray-400 mb-2 text-center">Quick Controls</div>
            <div className="flex justify-center gap-2 mb-4">
              <button
                onClick={() => setAngle(-15)}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg font-semibold"
              >
                ← Left
              </button>
              <button
                onClick={() => setAngle(0)}
                className="px-4 py-2 bg-green-500 hover:bg-green-600 rounded-lg font-semibold"
              >
                ↑ Straight
              </button>
              <button
                onClick={() => setAngle(15)}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 rounded-lg font-semibold"
              >
                Right →
              </button>
            </div>
          </div>

          <div>
            <div className="text-sm text-gray-400 mb-2 text-center">All Angles</div>
            <div className="grid grid-cols-7 gap-2">
              {STEERING_ANGLES.map((angle) => (
                <button
                  key={angle}
                  onClick={() => setAngle(angle)}
                  className={`px-3 py-2 rounded-lg font-semibold transition-colors ${
                    status.human_angle === angle
                      ? 'ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-800'
                      : getAngleColor(angle)
                  }`}
                >
                  {getAngleText(angle)}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Recording Controls */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">Recording Controls</h2>
          
          <div className="flex flex-wrap justify-center gap-4 mb-4">
            <button
              onClick={startRecording}
              disabled={status.is_recording}
              className="px-6 py-3 bg-red-500 hover:bg-red-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold"
            >
              Start Recording
            </button>
            <button
              onClick={stopRecording}
              disabled={!status.is_recording}
              className="px-6 py-3 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold"
            >
              Stop Recording
            </button>
          </div>
          
          <div className="text-center text-sm text-gray-400">
            {status.is_recording 
              ? 'Recording samples... Set steering angles using the controls above.'
              : 'Click "Start Recording" to begin collecting training samples.'}
          </div>
        </div>

        {/* Optimization Controls */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">Optimization Controls</h2>
          
          <div className="flex flex-wrap justify-center gap-4">
            <button
              onClick={runOptimization}
              disabled={isOptimizing || status.sample_count < 10}
              className="px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold"
            >
              {isOptimizing ? 'Optimizing...' : 'Run Optimization'}
            </button>
            <button
              onClick={saveWeights}
              disabled={isSaving || !status.optimization_result}
              className="px-6 py-3 bg-green-500 hover:bg-green-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold"
            >
              {isSaving ? 'Saving...' : 'Save Weights'}
            </button>
            <button
              onClick={resetSamples}
              className="px-6 py-3 bg-red-500 hover:bg-red-600 rounded-lg font-semibold"
            >
              Reset Samples
            </button>
          </div>

          {status.sample_count < 10 && (
            <div className="mt-4 text-center text-yellow-400 text-sm">
              Need at least 10 samples to run optimization. Current: {status.sample_count}
            </div>
          )}
        </div>

        {/* Optimization Results */}
        {status.optimization_result && (
          <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
            <h2 className="text-2xl font-bold mb-4 text-center">Optimization Results</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-yellow-400">Best Weights</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Far-Left / Far-Right:</span>
                    <span className="font-mono">{status.optimization_result.outer.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Left / Right:</span>
                    <span className="font-mono">{status.optimization_result.inner.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Center:</span>
                    <span className="font-mono">{status.optimization_result.center.toFixed(3)}</span>
                  </div>
                  <div className="border-t border-gray-600 pt-2 mt-2">
                    <div className="flex justify-between font-mono text-xs">
                      <span>Full Vector:</span>
                      <span className="text-right">
                        [{status.optimization_result.weights.map(w => w.toFixed(3)).join(', ')}]
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-green-400">Performance</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Mean Absolute Error:</span>
                    <span className="font-mono text-2xl font-bold text-yellow-400">
                      {status.optimization_result.mae.toFixed(2)}°
                    </span>
                  </div>
                  <div className="text-xs text-gray-400 mt-4">
                    Lower MAE indicates better prediction accuracy
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Camera Feeds - Side by Side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Camera Feed</h2>
            <CameraFeed />
          </div>

          <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Edge Detection</h2>
            <EdgeDetection />
          </div>
        </div>
      </div>
    </div>
  )
}

export default WeightTuner

