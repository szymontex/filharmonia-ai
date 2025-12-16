import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { CLASS_COLORS } from '../constants/colors'

interface Track {
  id: string
  predicted_class: string
  start: string
  stop: string
}

interface WaveformEditorProps {
  mp3Path: string
  tracks: Track[]
  onBack: () => void
}

function timeToSeconds(time: string): number {
  const parts = time.split(':').map(Number)
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

function secondsToTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

export default function WaveformEditor({ mp3Path, tracks, onBack }: WaveformEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState('00:00:00')
  const [duration, setDuration] = useState('00:00:00')
  const [loading, setLoading] = useState(true)
  const [waveformData, setWaveformData] = useState<any>(null)
  const [zoom, setZoom] = useState(1) // 1 = full view, 2 = 2x zoom, etc
  const [seeking, setSeeking] = useState(false)

  // Load waveform data
  useEffect(() => {
    const loadWaveform = async () => {
      try {
        const response = await axios.get(
          `/api/v1/waveform/data?path=${encodeURIComponent(mp3Path)}&samples_per_pixel=256`
        )
        setWaveformData(response.data)
        setLoading(false)
      } catch (error) {
        console.error('Error loading waveform:', error)
        setLoading(false)
      }
    }

    loadWaveform()
  }, [mp3Path])

  // Draw waveform
  useEffect(() => {
    if (!canvasRef.current || !waveformData) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const data = waveformData.data

    // Clear canvas with white background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform with darker color
    const step = width / data.length
    ctx.strokeStyle = '#4b5563' // gray-600 - darker
    ctx.lineWidth = 1

    ctx.beginPath()
    data.forEach((point: any, i: number) => {
      const x = i * step
      const yMin = (height / 2) - (point.min * height / 2)
      const yMax = (height / 2) - (point.max * height / 2)

      if (i === 0) {
        ctx.moveTo(x, yMin)
      }
      ctx.lineTo(x, yMin)
      ctx.lineTo(x, yMax)
    })
    ctx.stroke()

    // Draw regions (colored overlays)
    const totalDuration = waveformData.duration
    tracks.forEach(track => {
      const start = timeToSeconds(track.start)
      const end = timeToSeconds(track.stop)

      const startX = (start / totalDuration) * width
      const endX = (end / totalDuration) * width

      ctx.fillStyle = CLASS_COLORS[track.predicted_class as keyof typeof CLASS_COLORS]?.rgba || 'rgba(128, 128, 128, 0.4)'
      ctx.fillRect(startX, 0, endX - startX, height)
    })

    // Draw playhead
    if (audioRef.current) {
      const playheadX = (audioRef.current.currentTime / totalDuration) * width
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, 0)
      ctx.lineTo(playheadX, height)
      ctx.stroke()
    }

  }, [waveformData, tracks, zoom])

  // Redraw playhead on time update
  useEffect(() => {
    if (!audioRef.current) return

    const handleTimeUpdate = () => {
      if (!audioRef.current) return
      setCurrentTime(secondsToTime(audioRef.current.currentTime))

      // Redraw canvas with updated playhead
      if (canvasRef.current && waveformData) {
        // Trigger redraw
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Clear and redraw everything (simple approach)
        const width = canvas.width
        const height = canvas.height
        const data = waveformData.data

        ctx.fillStyle = '#ffffff'
        ctx.fillRect(0, 0, width, height)

        // Draw waveform
        const step = width / data.length
        ctx.strokeStyle = '#4b5563' // gray-600 - darker
        ctx.lineWidth = 1

        ctx.beginPath()
        data.forEach((point: any, i: number) => {
          const x = i * step
          const yMin = (height / 2) - (point.min * height / 2)
          const yMax = (height / 2) - (point.max * height / 2)

          if (i === 0) ctx.moveTo(x, yMin)
          ctx.lineTo(x, yMin)
          ctx.lineTo(x, yMax)
        })
        ctx.stroke()

        // Draw regions
        const totalDuration = waveformData.duration
        tracks.forEach(track => {
          const start = timeToSeconds(track.start)
          const end = timeToSeconds(track.stop)

          const startX = (start / totalDuration) * width
          const endX = (end / totalDuration) * width

          ctx.fillStyle = CLASS_COLORS[track.predicted_class as keyof typeof CLASS_COLORS]?.rgba || 'rgba(128, 128, 128, 0.4)'
          ctx.fillRect(startX, 0, endX - startX, height)
        })

        // Draw playhead
        const playheadX = (audioRef.current.currentTime / totalDuration) * width
        ctx.strokeStyle = '#ef4444'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(playheadX, 0)
        ctx.lineTo(playheadX, height)
        ctx.stroke()
      }
    }

    audioRef.current.addEventListener('timeupdate', handleTimeUpdate)
    return () => audioRef.current?.removeEventListener('timeupdate', handleTimeUpdate)
  }, [waveformData, tracks, zoom])

  const handlePlayPause = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !audioRef.current || !waveformData || seeking) return

    const audio = audioRef.current

    // Check if audio is ready
    if (audio.readyState < 2) {
      return
    }

    e.preventDefault()
    e.stopPropagation()

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left

    // Use canvas.width (actual canvas size) not rect.width (visible size)
    const clickRatio = x / canvas.width

    const newTime = Math.max(0, Math.min(audio.duration, clickRatio * audio.duration))

    // Pause, seek, then resume if was playing
    const wasPlaying = !audio.paused
    audio.pause()
    setSeeking(true)

    // Use seeked event to ensure seek completes
    const onSeeked = () => {
      audio.removeEventListener('seeked', onSeeked)
      setSeeking(false)
      if (wasPlaying) {
        audio.play()
      }
    }

    audio.addEventListener('seeked', onSeeked)
    audio.currentTime = newTime
  }

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.5, 10)) // Max 10x zoom
  }

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.5, 1)) // Min 1x (full view)
  }

  const handleExportMarkers = () => {
    const exported = tracks.map(t => ({
      id: t.id,
      start: t.start,
      end: t.stop,
      predicted_class: t.predicted_class
    }))

    console.log('Exported markers:', exported)
    alert(`Exported ${exported.length} markers! Check console for details.`)
  }

  const audioUrl = `/api/v1/audio/stream?path=${encodeURIComponent(mp3Path)}`

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <button
              onClick={onBack}
              className="text-blue-600 hover:text-blue-800 mb-2"
            >
              ← Back to CSV Viewer
            </button>
            <h1 className="text-3xl font-bold">Waveform Editor</h1>
            <p className="text-gray-600 text-sm mt-1">{mp3Path.split('\\').pop()}</p>
          </div>
          <button
            onClick={handleExportMarkers}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Export Markers
          </button>
        </div>

        {/* Waveform */}
        <div className="bg-white rounded-lg shadow p-6 mb-6 relative">
          {loading ? (
            <div className="text-center py-12 text-gray-500">
              Loading waveform data...
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <canvas
                  ref={canvasRef}
                  width={1200 * zoom}
                  height={200}
                  onClick={handleCanvasClick}
                  className="cursor-pointer border border-gray-200 rounded"
                  style={{ minWidth: '100%', opacity: seeking ? 0.5 : 1 }}
                />
              </div>
              {seeking && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10">
                  <div className="bg-white px-4 py-2 rounded shadow">
                    Seeking...
                  </div>
                </div>
              )}
            </>
          )}

          {/* Hidden audio element */}
          <audio
            ref={audioRef}
            src={audioUrl}
            preload="auto"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onLoadedMetadata={(e) => {
              const audio = e.target as HTMLAudioElement
              setDuration(secondsToTime(audio.duration))
              console.log('Audio loaded, duration:', audio.duration, 'readyState:', audio.readyState)
            }}
            onCanPlay={() => console.log('Audio can play, readyState:', audioRef.current?.readyState)}
          />
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={handlePlayPause}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium disabled:bg-gray-300"
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>

              <div className="text-sm text-gray-600">
                <span className="font-mono">{currentTime}</span>
                <span className="mx-2">/</span>
                <span className="font-mono">{duration}</span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500">Zoom: {zoom.toFixed(1)}x</span>
              <button
                onClick={handleZoomOut}
                disabled={zoom <= 1}
                className="px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
              >
                −
              </button>
              <button
                onClick={handleZoomIn}
                disabled={zoom >= 10}
                className="px-3 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
              >
                +
              </button>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Category Colors</h2>
          <div className="grid grid-cols-5 gap-4">
            {Object.entries(CLASS_COLORS).map(([cls, colorConfig]) => (
              <div key={cls} className="flex items-center gap-2">
                <div
                  className="w-8 h-8 rounded"
                  style={{ backgroundColor: colorConfig.rgba }}
                />
                <span className="text-sm font-medium">{cls}</span>
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-500 mt-4">
            💡 Tip: Click on waveform to seek audio position
          </p>
        </div>
      </div>
    </div>
  )
}
