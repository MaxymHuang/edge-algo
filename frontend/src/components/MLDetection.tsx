import { useEffect, useRef, useState } from 'react'

interface MLDetectionProps {
  modelType?: 'fast_scnn' | 'yolo'
}

const MLDetection = ({ modelType = 'fast_scnn' }: MLDetectionProps) => {
  const imgRef = useRef<HTMLImageElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const img = imgRef.current
    if (!img) return

    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
    const host = window.location.host
    img.src = `${protocol}//${host}/api/ml_feed?model_type=${modelType}`

    img.onerror = () => {
      setError(`Failed to load ML detection feed (${modelType})`)
    }

    return () => {
      img.src = ''
    }
  }, [modelType])

  if (error) {
    return (
      <div className="relative w-full bg-black rounded-lg overflow-hidden p-4">
        <p className="text-red-400 text-center">{error}</p>
      </div>
    )
  }

  return (
    <div className="relative w-full bg-black rounded-lg overflow-hidden">
      <img
        ref={imgRef}
        alt={`ML Detection (${modelType})`}
        className="w-full h-auto max-h-96 object-contain"
      />
    </div>
  )
}

export default MLDetection
