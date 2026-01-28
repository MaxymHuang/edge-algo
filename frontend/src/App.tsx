import { useState, useEffect } from 'react'
import CameraFeed from './components/CameraFeed'
import EdgeDetection from './components/EdgeDetection'
import MLDetection from './components/MLDetection'
import LogMessages from './components/LogMessages'

interface SteeringDetails {
  region_densities?: {
    far_left: number
    left: number
    center: number
    right: number
    far_right: number
  }
  weighted_densities?: {
    far_left: number
    left: number
    center: number
    right: number
    far_right: number
  }
  left_score?: number
  right_score?: number
  total_score?: number
  steering_score?: number
  total_density?: number
  // ML-related fields
  ml_confidence?: number
  ml_method?: string
  ml_angle?: number
  edge_angle?: number
  combined_angle?: number
  ml_weight?: number
  edge_weight?: number
  cached?: boolean
  error?: string
  method?: string
}

interface SteeringData {
  direction: string
  angle: number
  method?: string
  details?: SteeringDetails
}

function App() {
  const [steeringData, setSteeringData] = useState<SteeringData>({
    direction: 'straight',
    angle: 0
  })
  const [detectionMethod, setDetectionMethod] = useState<string>('edge')
  const [availableMethods] = useState<string[]>(['edge', 'fast_scnn', 'yolo', 'hybrid'])

  // Fetch current detection method on mount
  useEffect(() => {
    const fetchDetectionMethod = async () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
        const host = window.location.host
        const response = await fetch(`${protocol}//${host}/api/detection_method`)
        const data = await response.json()
        if (data.method) {
          setDetectionMethod(data.method)
        }
      } catch (error) {
        console.error('Error fetching detection method:', error)
      }
    }
    fetchDetectionMethod()
  }, [])

  // WebSocket for steering data
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/api/steering`)
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('Received steering data:', data)
        setSteeringData(data)
        if (data.method) {
          setDetectionMethod(data.method)
        }
      } catch (error) {
        console.error('Error parsing steering data:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    return () => {
      ws.close()
    }
  }, [])

  // Handle detection method change
  const handleMethodChange = async (method: string) => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
      const host = window.location.host
      const response = await fetch(`${protocol}//${host}/api/detection_method`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ method })
      })
      const data = await response.json()
      if (data.method) {
        setDetectionMethod(data.method)
      } else if (data.error) {
        console.error('Error setting detection method:', data.error)
        alert(`Failed to set detection method: ${data.error}`)
      }
    } catch (error) {
      console.error('Error setting detection method:', error)
      alert(`Failed to set detection method: ${error}`)
    }
  }

  const getDirectionColor = () => {
    switch (steeringData.direction) {
      case 'left':
        return 'bg-blue-500'
      case 'right':
        return 'bg-red-500'
      default:
        return 'bg-green-500'
    }
  }

  const getDirectionText = () => {
    switch (steeringData.direction) {
      case 'left':
        return 'LEFT'
      case 'right':
        return 'RIGHT'
      default:
        return 'STRAIGHT'
    }
  }

  const formatPercentage = (value: number): string => {
    return (value * 100).toFixed(2) + '%'
  }

  const formatDecimal = (value: number, decimals: number = 4): string => {
    return value.toFixed(decimals)
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-center">Vision Algorithm Console</h1>
        
        {/* Detection Method Selection */}
        <div className="mb-6 bg-gray-800 rounded-lg p-4 shadow-lg">
          <h2 className="text-xl font-semibold mb-3">Detection Method</h2>
          <div className="flex items-center gap-4">
            <label htmlFor="method-select" className="text-sm font-medium">
              Method:
            </label>
            <select
              id="method-select"
              value={detectionMethod}
              onChange={(e) => handleMethodChange(e.target.value)}
              className="bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableMethods.map((method) => (
                <option key={method} value={method}>
                  {method.charAt(0).toUpperCase() + method.slice(1)}
                </option>
              ))}
            </select>
            <span className="text-sm text-gray-400">
              Current: <span className="font-mono text-yellow-400">{detectionMethod}</span>
            </span>
          </div>
        </div>
        
        {/* Steering Control Panel */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">
            Steering Control Panel
            {steeringData.method && (
              <span className="text-sm font-normal text-gray-400 ml-2">
                ({steeringData.method})
              </span>
            )}
          </h2>
          
          {/* Main Steering Indicator */}
          <div className="mb-6 flex justify-center">
            <div className={`${getDirectionColor()} px-8 py-4 rounded-lg text-2xl font-bold shadow-lg`}>
              Steering: {getDirectionText()} ({steeringData.angle > 0 ? '+' : ''}{steeringData.angle}°)
            </div>
          </div>

          {/* Detailed Calculation Results */}
          {steeringData.details ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Region Densities */}
              {steeringData.details.region_densities && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-yellow-400">Region Densities</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Far-Left:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.region_densities.far_left)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Left:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.region_densities.left)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Center:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.region_densities.center)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Right:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.region_densities.right)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Far-Right:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.region_densities.far_right)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Weighted Densities */}
              {steeringData.details.weighted_densities && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-semibold mb-3 text-green-400">Weighted Densities</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Far-Left:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.weighted_densities.far_left)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Left:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.weighted_densities.left)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Center:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.weighted_densities.center)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Right:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.weighted_densities.right)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Far-Right:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.weighted_densities.far_right)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Scores and Final Value */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">Scores & Final Value</h3>
                <div className="space-y-2 text-sm">
                  {steeringData.details.ml_confidence !== undefined && (
                    <>
                      <div className="flex justify-between">
                        <span>ML Confidence:</span>
                        <span className="font-mono text-green-400">{formatPercentage(steeringData.details.ml_confidence)}</span>
                      </div>
                      {steeringData.details.ml_method && (
                        <div className="flex justify-between">
                          <span>ML Method:</span>
                          <span className="font-mono text-green-400">{steeringData.details.ml_method}</span>
                        </div>
                      )}
                      {steeringData.details.cached !== undefined && (
                        <div className="flex justify-between">
                          <span>Cached:</span>
                          <span className="font-mono">{steeringData.details.cached ? 'Yes' : 'No'}</span>
                        </div>
                      )}
                      <div className="border-t border-gray-600 pt-2 mt-2"></div>
                    </>
                  )}
                  {steeringData.details.left_score !== undefined && (
                    <>
                      <div className="flex justify-between">
                        <span>Left Score:</span>
                        <span className="font-mono">{formatDecimal(steeringData.details.left_score)}</span>
                      </div>
                      {steeringData.details.right_score !== undefined && (
                        <div className="flex justify-between">
                          <span>Right Score:</span>
                          <span className="font-mono">{formatDecimal(steeringData.details.right_score)}</span>
                        </div>
                      )}
                      {steeringData.details.total_score !== undefined && (
                        <div className="flex justify-between">
                          <span>Total Score:</span>
                          <span className="font-mono">{formatDecimal(steeringData.details.total_score)}</span>
                        </div>
                      )}
                    </>
                  )}
                  {steeringData.details.steering_score !== undefined && (
                    <div className="border-t border-gray-600 pt-2 mt-2">
                      <div className="flex justify-between font-semibold">
                        <span>Steering Score:</span>
                        <span className="font-mono">{formatDecimal(steeringData.details.steering_score)}</span>
                      </div>
                    </div>
                  )}
                  {steeringData.details.total_density !== undefined && (
                    <div className="flex justify-between">
                      <span>Total Density:</span>
                      <span className="font-mono">{formatPercentage(steeringData.details.total_density)}</span>
                    </div>
                  )}
                  {steeringData.details.ml_angle !== undefined && steeringData.details.edge_angle !== undefined && (
                    <>
                      <div className="border-t border-gray-600 pt-2 mt-2"></div>
                      <div className="flex justify-between">
                        <span>ML Angle:</span>
                        <span className="font-mono text-green-400">{steeringData.details.ml_angle}°</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Edge Angle:</span>
                        <span className="font-mono text-blue-400">{steeringData.details.edge_angle}°</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ML Weight:</span>
                        <span className="font-mono">{formatPercentage(steeringData.details.ml_weight || 0)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Edge Weight:</span>
                        <span className="font-mono">{formatPercentage(steeringData.details.edge_weight || 0)}</span>
                      </div>
                    </>
                  )}
                  <div className="border-t border-gray-600 pt-2 mt-2">
                    <div className="flex justify-between font-bold text-lg">
                      <span>Final Angle:</span>
                      <span className="font-mono text-yellow-400">{steeringData.angle > 0 ? '+' : ''}{steeringData.angle}°</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-4">
              <p>Waiting for steering calculation data...</p>
              <p className="text-xs mt-2 text-gray-500">
                Current data: {JSON.stringify(steeringData, null, 2)}
              </p>
            </div>
          )}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Camera Feed */}
          <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Camera Feed</h2>
            <CameraFeed />
          </div>

          {/* Edge Detection or ML Detection */}
          {detectionMethod === 'edge' || detectionMethod === 'hybrid' ? (
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
              <h2 className="text-xl font-semibold mb-4">Edge Detection</h2>
              <EdgeDetection />
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
              <h2 className="text-xl font-semibold mb-4">
                ML Detection ({detectionMethod === 'fast_scnn' ? 'Fast SCNN' : 'YOLOv8'})
              </h2>
              <MLDetection modelType={detectionMethod === 'fast_scnn' ? 'fast_scnn' : 'yolo'} />
            </div>
          )}
        </div>

        {/* ML Detection (if hybrid mode) */}
        {detectionMethod === 'hybrid' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
              <h2 className="text-xl font-semibold mb-4">ML Detection (Fast SCNN)</h2>
              <MLDetection modelType="fast_scnn" />
            </div>
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
              <h2 className="text-xl font-semibold mb-4">ML Detection (YOLOv8)</h2>
              <MLDetection modelType="yolo" />
            </div>
          </div>
        )}

        {/* Log Messages */}
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
          <h2 className="text-xl font-semibold mb-4">Log Messages</h2>
          <LogMessages />
        </div>
      </div>
    </div>
  )
}

export default App

