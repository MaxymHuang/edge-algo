import { useEffect, useRef } from 'react'

const CameraFeed = () => {
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const img = imgRef.current
    if (!img) return

    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'
    const host = window.location.host
    img.src = `${protocol}//${host}/api/video_feed`

    return () => {
      img.src = ''
    }
  }, [])

  return (
    <div className="relative w-full bg-black rounded-lg overflow-hidden">
      <img
        ref={imgRef}
        alt="Camera Feed"
        className="w-full h-auto max-h-96 object-contain"
      />
    </div>
  )
}

export default CameraFeed

