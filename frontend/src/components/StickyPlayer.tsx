import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { CLASS_COLORS } from '../constants/colors'

interface Track {
  id: string
  predicted_class: string
  start: string
  stop: string
}

interface StickyPlayerProps {
  mp3Path: string
  tracks: Track[]
  onClose: () => void
  onTrackUpdate?: (trackId: string, updates: { start?: string; stop?: string }) => void
  onBoundaryUpdate?: (prevTrackId: string, nextTrackId: string, time: string) => void
  selectedTrackId?: string | null
  onTrackSelect?: (trackId: string) => void
  seekToTime?: string | null
  onSeekComplete?: () => void
  recordingDate?: string | null
  onAddSegment?: (timeStr: string, totalDuration?: number) => void
  onCutSegment?: (timeStr: string) => void
  onPlayingTrackChange?: (trackId: string | null) => void
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

export default function StickyPlayer({ mp3Path, tracks, onClose, onTrackUpdate, onBoundaryUpdate, selectedTrackId, onTrackSelect, seekToTime, onSeekComplete, recordingDate, onAddSegment, onCutSegment, onPlayingTrackChange }: StickyPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState('00:00:00')
  const [duration, setDuration] = useState('00:00:00')
  const [loading, setLoading] = useState(true)
  const [waveformData, setWaveformData] = useState<any>(null)
  const [zoom, setZoom] = useState(1)
  const [amplitudeScale, setAmplitudeScale] = useState(1)  // Vertical zoom for waveform amplitude
  const [seeking, setSeeking] = useState(false)
  const [draggingMarker, setDraggingMarker] = useState<{ prevTrackId: string; nextTrackId: string } | null>(null)
  const [hoverMarker, setHoverMarker] = useState<{ prevTrackId: string; nextTrackId: string } | null>(null)
  const [addMode, setAddMode] = useState(false)

  // Load waveform data
  useEffect(() => {
    const loadWaveform = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/v1/waveform/data?path=${encodeURIComponent(mp3Path)}&samples_per_pixel=256`
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

  // Pause and reset audio when tracks change (e.g., switching CSV)
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setIsPlaying(false)
      setCurrentTime('00:00:00')
    }
  }, [tracks])

  // Handle wheel zoom with native event listener (to use preventDefault)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleNativeWheel = (e: WheelEvent) => {
      if (!scrollContainerRef.current || !waveformData) return

      e.preventDefault() // This works with passive: false

      const container = scrollContainerRef.current
      const containerRect = container.getBoundingClientRect()

      // Mouse position relative to CONTAINER (viewport), not canvas
      const mouseXInContainer = e.clientX - containerRect.left

      // Calculate OLD canvas width from current zoom
      const oldCanvasWidth = 1400 * zoom

      // Get mouse position in canvas coordinates (accounting for scroll)
      const canvasMouseX = mouseXInContainer + container.scrollLeft

      // Calculate what percentage of the canvas the mouse is at
      const mouseRatio = canvasMouseX / oldCanvasWidth

      // Zoom in or out based on wheel direction
      const delta = e.deltaY > 0 ? -1 : 1
      let newZoom = zoom

      if (delta > 0) {
        newZoom = Math.min(zoom * 1.5, 10)
      } else {
        newZoom = Math.max(zoom / 1.5, 1)
      }

      if (newZoom === zoom) return

      // Calculate NEW canvas width
      const newCanvasWidth = 1400 * newZoom

      // Mouse should stay at the same ratio position
      const newCanvasMouseX = newCanvasWidth * mouseRatio

      // Calculate new scroll position to keep mouse at same position IN CONTAINER
      const newScrollLeft = newCanvasMouseX - mouseXInContainer

      setZoom(newZoom)

      // Use requestAnimationFrame for smoother update
      requestAnimationFrame(() => {
        container.scrollLeft = newScrollLeft
      })
    }

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleNativeWheel)
  }, [zoom, waveformData])

  // Handle external seek requests
  useEffect(() => {
    if (!seekToTime || !audioRef.current || !waveformData || !scrollContainerRef.current || !canvasRef.current) return

    const audio = audioRef.current
    if (audio.readyState < 2) return // Wait for audio to be ready

    const seekSeconds = timeToSeconds(seekToTime)
    audio.currentTime = seekSeconds
    audio.play()

    // Auto-scroll waveform to show the playhead
    const container = scrollContainerRef.current
    const canvas = canvasRef.current
    const progress = seekSeconds / audio.duration
    const playheadX = progress * canvas.width

    // Center the playhead in the viewport
    const scrollLeft = playheadX - container.clientWidth / 2
    container.scrollTo({
      left: Math.max(0, scrollLeft),
      behavior: 'smooth'
    })

    // Notify parent that seek is complete
    if (onSeekComplete) {
      onSeekComplete()
    }
  }, [seekToTime, waveformData, onSeekComplete])

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
      const yMin = (height / 2) - (point.min * height / 2 * amplitudeScale)
      const yMax = (height / 2) - (point.max * height / 2 * amplitudeScale)

      if (i === 0) {
        ctx.moveTo(x, yMin)
      }
      ctx.lineTo(x, yMin)
      ctx.lineTo(x, yMax)
    })
    ctx.stroke()

    // Draw regions (colored overlays)
    const totalDuration = waveformData.duration
    tracks.forEach((track, idx) => {
      const start = timeToSeconds(track.start)
      const startX = (start / totalDuration) * width

      // End position: if there's a next track, draw until its start; otherwise use track's stop
      let endX
      if (idx < tracks.length - 1) {
        const nextTrackStart = timeToSeconds(tracks[idx + 1].start)
        endX = (nextTrackStart / totalDuration) * width
      } else {
        const end = timeToSeconds(track.stop)
        endX = (end / totalDuration) * width
      }

      ctx.fillStyle = CLASS_COLORS[track.predicted_class as keyof typeof CLASS_COLORS]?.rgba || 'rgba(128, 128, 128, 0.4)'
      ctx.fillRect(startX, 0, endX - startX, height)

      // Draw selection border if this track is selected (inset to avoid overlapping markers)
      if (selectedTrackId === track.id) {
        ctx.strokeStyle = '#2563eb' // blue-600
        ctx.lineWidth = 3
        // Inset by 1.5px (half the line width) to keep border inside the region
        ctx.strokeRect(startX + 1.5, 1.5, (endX - startX) - 3, height - 3)
      }
    })

    // Draw boundary markers at the start of each next track
    for (let i = 0; i < tracks.length - 1; i++) {
      const currentTrack = tracks[i]
      const nextTrack = tracks[i + 1]

      // Marker at the start of the next track
      const nextStart = timeToSeconds(nextTrack.start)
      const boundaryX = (nextStart / totalDuration) * width

      // Check if this boundary is being hovered
      const isHover = hoverMarker?.prevTrackId === currentTrack.id && hoverMarker?.nextTrackId === nextTrack.id

      ctx.strokeStyle = isHover ? '#3b82f6' : '#1f2937'
      ctx.lineWidth = isHover ? 3 : 2
      ctx.beginPath()
      ctx.moveTo(boundaryX, 0)
      ctx.lineTo(boundaryX, height)
      ctx.stroke()
    }

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

  }, [waveformData, tracks, zoom, hoverMarker, selectedTrackId, amplitudeScale])

  // Redraw playhead on time update
  useEffect(() => {
    if (!audioRef.current) return

    const handleTimeUpdate = () => {
      if (!audioRef.current) return
      const currentSeconds = audioRef.current.currentTime
      setCurrentTime(secondsToTime(currentSeconds))

      // Find which track is currently playing
      if (onPlayingTrackChange) {
        const playingTrack = tracks.find(track => {
          const startSeconds = timeToSeconds(track.start)
          const stopSeconds = timeToSeconds(track.stop)
          return currentSeconds >= startSeconds && currentSeconds < stopSeconds
        })
        onPlayingTrackChange(playingTrack ? playingTrack.id : null)
      }

      // Redraw canvas with updated playhead
      if (canvasRef.current && waveformData) {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

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
          const yMin = (height / 2) - (point.min * height / 2 * amplitudeScale)
          const yMax = (height / 2) - (point.max * height / 2 * amplitudeScale)

          if (i === 0) ctx.moveTo(x, yMin)
          ctx.lineTo(x, yMin)
          ctx.lineTo(x, yMax)
        })
        ctx.stroke()

        // Draw regions
        const totalDuration = waveformData.duration
        tracks.forEach((track, idx) => {
          const start = timeToSeconds(track.start)
          const startX = (start / totalDuration) * width

          // End position: if there's a next track, draw until its start; otherwise use track's stop
          let endX
          if (idx < tracks.length - 1) {
            const nextTrackStart = timeToSeconds(tracks[idx + 1].start)
            endX = (nextTrackStart / totalDuration) * width
          } else {
            const end = timeToSeconds(track.stop)
            endX = (end / totalDuration) * width
          }

          ctx.fillStyle = CLASS_COLORS[track.predicted_class as keyof typeof CLASS_COLORS]?.rgba || 'rgba(128, 128, 128, 0.4)'
          ctx.fillRect(startX, 0, endX - startX, height)

          // Draw selection border if this track is selected (inset to avoid overlapping markers)
          if (selectedTrackId === track.id) {
            ctx.strokeStyle = '#2563eb' // blue-600
            ctx.lineWidth = 3
            // Inset by 1.5px (half the line width) to keep border inside the region
            ctx.strokeRect(startX + 1.5, 1.5, (endX - startX) - 3, height - 3)
          }
        })

        // Draw boundary markers
        for (let i = 0; i < tracks.length - 1; i++) {
          const currentTrack = tracks[i]
          const nextTrack = tracks[i + 1]

          // Marker at the start of the next track
          const nextStart = timeToSeconds(nextTrack.start)
          const boundaryX = (nextStart / totalDuration) * width

          const isHover = hoverMarker?.prevTrackId === currentTrack.id && hoverMarker?.nextTrackId === nextTrack.id

          ctx.strokeStyle = isHover ? '#3b82f6' : '#1f2937'
          ctx.lineWidth = isHover ? 3 : 2
          ctx.beginPath()
          ctx.moveTo(boundaryX, 0)
          ctx.lineTo(boundaryX, height)
          ctx.stroke()
        }

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
  }, [waveformData, tracks, zoom, selectedTrackId, amplitudeScale])

  const handlePlayPause = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
  }

  // Helper to find boundary marker between consecutive tracks
  const findMarkerNear = (canvasX: number): { prevTrackId: string; nextTrackId: string } | null => {
    if (!canvasRef.current || !waveformData) return null

    const canvas = canvasRef.current
    const width = canvas.width
    const totalDuration = waveformData.duration
    const threshold = 10 // pixels

    // Find boundaries between consecutive tracks
    for (let i = 0; i < tracks.length - 1; i++) {
      const currentTrack = tracks[i]
      const nextTrack = tracks[i + 1]

      // Marker at the start of the next track
      const nextStart = timeToSeconds(nextTrack.start)
      const boundaryX = (nextStart / totalDuration) * width

      if (Math.abs(canvasX - boundaryX) < threshold) {
        return { prevTrackId: currentTrack.id, nextTrackId: nextTrack.id }
      }
    }

    return null
  }

  // Helper to find which track was clicked/hovered
  const findTrackAtPosition = (canvasX: number): string | null => {
    if (!canvasRef.current || !waveformData) return null

    const canvas = canvasRef.current
    const width = canvas.width
    const totalDuration = waveformData.duration
    const clickTime = (canvasX / width) * totalDuration

    for (let i = 0; i < tracks.length; i++) {
      const track = tracks[i]
      const start = timeToSeconds(track.start)

      // End is either next track's start or current track's stop
      let end
      if (i < tracks.length - 1) {
        end = timeToSeconds(tracks[i + 1].start)
      } else {
        end = timeToSeconds(track.stop)
      }

      if (clickTime >= start && clickTime <= end) {
        return track.id
      }
    }

    return null
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !waveformData) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left

    if (draggingMarker) {
      // Update boundary marker position during drag - affects both adjacent segments
      const canvas = canvasRef.current
      const ratio = x / canvas.width
      const newTime = Math.max(0, Math.min(waveformData.duration, ratio * waveformData.duration))
      const newTimeStr = secondsToTime(newTime)

      // Use boundary update if available (updates both segments atomically)
      if (onBoundaryUpdate) {
        onBoundaryUpdate(draggingMarker.prevTrackId, draggingMarker.nextTrackId, newTimeStr)
      } else if (onTrackUpdate) {
        // Fallback to individual updates
        onTrackUpdate(draggingMarker.prevTrackId, { stop: newTimeStr })
        onTrackUpdate(draggingMarker.nextTrackId, { start: newTimeStr })
      }
    } else {
      // Update hover state for markers
      const marker = findMarkerNear(x)
      setHoverMarker(marker)

      // Update hover state for track selection
      const hoveredTrackId = findTrackAtPosition(x)
      if (hoveredTrackId && onTrackSelect) {
        onTrackSelect(hoveredTrackId)
      }
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left

    const marker = findMarkerNear(x)
    if (marker) {
      e.preventDefault()
      e.stopPropagation()
      setDraggingMarker(marker)
    }
  }

  const handleMouseUp = () => {
    setDraggingMarker(null)
  }

  const handleMouseLeave = () => {
    setHoverMarker(null)
    setDraggingMarker(null)
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !audioRef.current || !waveformData || seeking) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left

    // Don't seek if we clicked on a marker
    if (draggingMarker || hoverMarker) return

    const audio = audioRef.current

    if (audio.readyState < 2) return

    e.preventDefault()
    e.stopPropagation()

    const clickRatio = x / canvas.width
    const clickTimeSeconds = clickRatio * audio.duration

    // If in add mode OR holding Ctrl/Cmd, add segment instead of seeking
    if ((addMode || e.ctrlKey || e.metaKey) && onAddSegment) {
      const clickTimeStr = secondsToTime(clickTimeSeconds)
      onAddSegment(clickTimeStr, audio.duration)
      return
    }

    // Normal seek behavior
    const newTime = Math.max(0, Math.min(audio.duration, clickTimeSeconds))

    const wasPlaying = !audio.paused
    audio.pause()
    setSeeking(true)

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

  const audioUrl = `http://localhost:8000/api/v1/audio/stream?path=${encodeURIComponent(mp3Path)}`

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t-2 border-gray-300 shadow-2xl z-50">
      <div className="max-w-7xl mx-auto p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h3 className="font-semibold text-lg">Audio Player</h3>
            <span className="text-sm text-gray-600">{mp3Path.split('\\').pop()?.replace('.MP3', '').replace('.mp3', '')}</span>
            {recordingDate && (
              <>
                <span className="text-gray-400">•</span>
                <span className="text-sm text-blue-600 font-medium">
                  {(() => {
                    const [year, month, day] = recordingDate.split('-')
                    return `${day}.${month}.${year}`
                  })()}
                </span>
              </>
            )}
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Waveform */}
        <div className="mb-3 relative flex gap-2">
          {loading ? (
            <div className="text-center py-8 text-gray-500 text-sm flex-1">
              Loading waveform...
            </div>
          ) : (
            <>
              <div ref={scrollContainerRef} className="overflow-x-auto flex-1">
                <canvas
                  ref={canvasRef}
                  width={1400 * zoom}
                  height={150}
                  onClick={handleCanvasClick}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseLeave}
                  className="cursor-pointer border border-gray-200 rounded"
                  style={{
                    minWidth: '100%',
                    opacity: seeking ? 0.5 : 1,
                    cursor: hoverMarker ? 'ew-resize' : (addMode ? 'crosshair' : 'pointer')
                  }}
                />
              </div>

              {/* Vertical amplitude slider */}
              <div className="flex flex-col items-center gap-1 bg-gray-50 px-2 py-2 rounded border border-gray-200">
                <span className="text-xs text-gray-600 font-medium whitespace-nowrap" title="Waveform amplitude zoom">
                  Amp
                </span>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={amplitudeScale}
                  onChange={(e) => setAmplitudeScale(parseFloat(e.target.value))}
                  className="h-24 cursor-pointer"
                  style={{
                    writingMode: 'bt-lr',
                    WebkitAppearance: 'slider-vertical',
                    width: '8px'
                  }}
                  title={`Amplitude: ${amplitudeScale}x`}
                />
                <span className="text-xs text-blue-600 font-semibold">
                  {amplitudeScale}x
                </span>
              </div>

              {seeking && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10">
                  <div className="bg-white px-3 py-1 rounded shadow text-sm">
                    Seeking...
                  </div>
                </div>
              )}
            </>
          )}

          <audio
            ref={audioRef}
            src={audioUrl}
            preload="auto"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onLoadedMetadata={(e) => {
              const audio = e.target as HTMLAudioElement
              setDuration(secondsToTime(audio.duration))
            }}
          />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={handlePlayPause}
              disabled={loading}
              className="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium disabled:bg-gray-300"
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>

            {onCutSegment && audioRef.current && (
              <button
                onClick={() => {
                  const currentTimeStr = secondsToTime(audioRef.current?.currentTime || 0)
                  onCutSegment(currentTimeStr)
                }}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium disabled:bg-gray-300"
                title="Split segment at playhead (keeps same class)"
              >
                ✂️ Cut
              </button>
            )}

            {onAddSegment && audioRef.current && (
              <button
                onClick={() => {
                  const currentTimeStr = secondsToTime(audioRef.current?.currentTime || 0)
                  onAddSegment(currentTimeStr, audioRef.current?.duration)
                }}
                disabled={loading}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 font-medium disabled:bg-gray-300"
                title="Add new segment at playhead (8 seconds)"
              >
                + Add
              </button>
            )}

            <div className="text-sm text-gray-600">
              <span className="font-mono">{currentTime}</span>
              <span className="mx-1">/</span>
              <span className="font-mono">{duration}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {onAddSegment && (
              <button
                onClick={() => setAddMode(!addMode)}
                className={`px-3 py-1 text-sm rounded font-medium ${
                  addMode
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                title="Toggle add segment mode (or hold Ctrl/Cmd while clicking)"
              >
                {addMode ? '✓ Add Mode ON' : '+ Add Mode'}
              </button>
            )}
            <span className="text-sm text-gray-500">Zoom: {zoom.toFixed(1)}x (scroll to zoom)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
