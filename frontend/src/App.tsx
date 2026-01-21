import { useState, useEffect } from 'react'
import CameraFeed from './components/CameraFeed'
import EdgeDetection from './components/EdgeDetection'
import LogMessages from './components/LogMessages'

interface SteeringDetails {
  region_densities: {
    far_left: number
    left: number
    center: number
    right: number
    far_right: number
  }
  weighted_densities: {
    far_left: number
    left: number
    center: number
    right: number
    far_right: number
  }
  left_score: number
  right_score: number
  total_score: number
  steering_score: number
  total_density: number
}

interface SteeringData {
  direction: string
  angle: number
  details?: SteeringDetails
}

function App() {
  const [steeringData, setSteeringData] = useState<SteeringData>({
    direction: 'straight',
    angle: 0
  })

  useEffect(() => {
    const ws = new WebSocket(`ws://${window.location.host}/api/steering`)
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('Received steering data:', data)
        setSteeringData(data)
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
        
        {/* Steering Control Panel */}
        <div className="mb-6 bg-gray-800 rounded-lg p-6 shadow-lg">
          <h2 className="text-2xl font-bold mb-4 text-center">Steering Control Panel</h2>
          
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

              {/* Weighted Densities */}
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

              {/* Scores and Final Value */}
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3 text-blue-400">Scores & Final Value</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Left Score:</span>
                    <span className="font-mono">{formatDecimal(steeringData.details.left_score)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Right Score:</span>
                    <span className="font-mono">{formatDecimal(steeringData.details.right_score)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Score:</span>
                    <span className="font-mono">{formatDecimal(steeringData.details.total_score)}</span>
                  </div>
                  <div className="border-t border-gray-600 pt-2 mt-2">
                    <div className="flex justify-between font-semibold">
                      <span>Steering Score:</span>
                      <span className="font-mono">{formatDecimal(steeringData.details.steering_score)}</span>
                    </div>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Density:</span>
                    <span className="font-mono">{formatPercentage(steeringData.details.total_density)}</span>
                  </div>
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

          {/* Edge Detection */}
          <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Edge Detection</h2>
            <EdgeDetection />
          </div>
        </div>

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

