import { useEffect, useState, useRef } from 'react'

interface LogEntry {
  timestamp: number
  level: string
  message: string
}

const LogMessages = () => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const logContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const ws = new WebSocket(`${protocol}//${host}/api/logs`)
    
    ws.onmessage = (event) => {
      const logEntry: LogEntry = JSON.parse(event.data)
      setLogs((prevLogs) => {
        const newLogs = [...prevLogs, logEntry]
        // Keep only last 500 logs
        return newLogs.slice(-500)
      })
    }

    ws.onerror = (error) => {
      console.error('Log WebSocket error:', error)
    }

    // Fetch initial logs
    fetch('/api/logs')
      .then((res) => res.json())
      .then((data) => {
        if (data.logs) {
          setLogs(data.logs)
        }
      })
      .catch((err) => {
        console.error('Failed to fetch initial logs:', err)
      })

    return () => {
      ws.close()
    }
  }, [])

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const getLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'text-red-400'
      case 'WARNING':
        return 'text-yellow-400'
      case 'INFO':
        return 'text-blue-400'
      default:
        return 'text-gray-300'
    }
  }

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000)
    return date.toLocaleTimeString()
  }

  return (
    <div
      ref={logContainerRef}
      className="bg-black rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm"
    >
      {logs.length === 0 ? (
        <div className="text-gray-500">No log messages yet...</div>
      ) : (
        logs.map((log, index) => (
          <div key={index} className="mb-1">
            <span className="text-gray-500">[{formatTimestamp(log.timestamp)}]</span>
            <span className={`ml-2 ${getLevelColor(log.level)}`}>
              [{log.level}]
            </span>
            <span className="ml-2 text-gray-300">{log.message}</span>
          </div>
        ))
      )}
    </div>
  )
}

export default LogMessages

