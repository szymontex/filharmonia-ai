import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { CLASS_COLORS } from '../constants/colors'
import Toast from '../components/Toast'

interface UncertainSegment {
  csv_path: string
  mp3_path: string
  segment_index: number
  segment_time: string
  predicted_class: string
  confidence: number
}

interface Stats {
  csvs_with_confidence: number
  total_uncertain_segments: number
  total_reviewed: number
  remaining: number
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

export default function UncertaintyReview({ onBack }: { onBack: () => void }) {
  const [segments, setSegments] = useState<UncertainSegment[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(true)
  const [loadingWaveform, setLoadingWaveform] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0)
  const [maxConfidence, setMaxConfidence] = useState(0.7)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)

  // Waveform and audio
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [waveformData, setWaveformData] = useState<any>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState('00:00:00')
  const [duration, setDuration] = useState('00:00:00')

  // Range selection
  const [rangeStart, setRangeStart] = useState<number | null>(null)
  const [rangeEnd, setRangeEnd] = useState<number | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Amplitude zoom
  const [amplitudeScale, setAmplitudeScale] = useState(1)

  // Horizontal zoom
  const [zoom, setZoom] = useState(1)
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  // Debug zoom changes
  useEffect(() => {
    console.log('[UncertaintyReview] Zoom changed to:', zoom)
  }, [zoom])

  // Toast
  const [successToast, setSuccessToast] = useState({ show: false, message: '' })
  const [errorToast, setErrorToast] = useState({ show: false, message: '' })

  const currentSegment = segments[currentIndex]

  // Debounce filter changes to avoid too many requests
  useEffect(() => {
    const timer = setTimeout(() => {
      loadUncertainSegments()
      loadStats()
    }, 500) // Wait 500ms after user stops moving slider

    return () => clearTimeout(timer)
  }, [minConfidence, maxConfidence, selectedCategory])

  const loadUncertainSegments = async () => {
    try {
      setLoading(true)
      const params = new URLSearchParams({
        min_confidence: minConfidence.toString(),
        max_confidence: maxConfidence.toString(),
        limit: '50'
      })
      if (selectedCategory) {
        params.append('category', selectedCategory)
      }
      const res = await axios.get(`/api/v1/uncertainty/segments?${params}`)
      setSegments(res.data.segments)
      setLoading(false)
    } catch (error: any) {
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to load segments' })
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const res = await axios.get('/api/v1/uncertainty/stats')
      setStats(res.data)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  // Handle wheel zoom (horizontal)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleNativeWheel = (e: WheelEvent) => {
      if (!scrollContainerRef.current || !waveformData) return

      e.preventDefault()

      const container = scrollContainerRef.current
      const containerRect = container.getBoundingClientRect()

      // Mouse position relative to CONTAINER (viewport), not canvas
      const mouseXInContainer = e.clientX - containerRect.left

      // Calculate OLD canvas width from current zoom
      const oldCanvasWidth = 1000 * zoom

      // Get mouse position in canvas coordinates (accounting for scroll)
      const canvasMouseX = mouseXInContainer + container.scrollLeft

      // Calculate what percentage of the canvas the mouse is at
      const mouseRatio = canvasMouseX / oldCanvasWidth

      // Zoom in or out based on wheel direction
      const delta = e.deltaY > 0 ? -1 : 1
      let newZoom = zoom

      if (delta > 0) {
        newZoom = Math.min(zoom * 1.5, 50)
      } else {
        newZoom = Math.max(zoom / 1.5, 1)
      }

      if (newZoom === zoom) return

      // Calculate new canvas width and scroll position
      const newCanvasWidth = 1000 * newZoom
      const newCanvasMouseX = newCanvasWidth * mouseRatio
      const newScrollLeft = newCanvasMouseX - mouseXInContainer

      setZoom(newZoom)

      requestAnimationFrame(() => {
        container.scrollLeft = newScrollLeft
      })
    }

    canvas.addEventListener('wheel', handleNativeWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleNativeWheel)
  }, [zoom, waveformData])

  // Load waveform when segment changes
  useEffect(() => {
    if (!currentSegment) return

    console.log('[UncertaintyReview] loadWaveform triggered for segment:', currentSegment.segment_index, currentSegment.segment_time)

    const loadWaveform = async () => {
      setLoadingWaveform(true)
      setIsPlaying(false)
      try {
        // Load with fewer samples = faster (512 instead of 256)
        const response = await axios.get(
          `/api/v1/waveform/data?path=${encodeURIComponent(currentSegment.mp3_path)}&samples_per_pixel=512`
        )

        // Calculate segment position BEFORE setting state
        const segmentStartSec = timeToSeconds(currentSegment.segment_time)
        const segmentEndSec = segmentStartSec + 2.97  // FRAME_DURATION_SEC
        const segmentCenterSec = (segmentStartSec + segmentEndSec) / 2
        const startSec = Math.max(0, segmentCenterSec - 10)
        const endSec = Math.min(response.data.duration, segmentCenterSec + 10)

        // Set all state at once to minimize re-renders
        setWaveformData(response.data)
        setRangeStart(startSec)
        setRangeEnd(endSec)
        setDuration(secondsToTime(response.data.duration))

        // Scroll to segment center after render
        requestAnimationFrame(() => {
          if (scrollContainerRef.current && canvasRef.current) {
            const canvasWidth = canvasRef.current.width
            const segmentRatio = segmentCenterSec / response.data.duration
            const segmentX = canvasWidth * segmentRatio
            const containerWidth = scrollContainerRef.current.clientWidth
            scrollContainerRef.current.scrollLeft = segmentX - (containerWidth / 2)
          }
          setLoadingWaveform(false)
          setExporting(false)
        })
      } catch (error) {
        console.error('Error loading waveform:', error)
        setLoadingWaveform(false)
        setExporting(false)  // Clear exporting flag on error too
      }
    }

    loadWaveform()
  }, [currentSegment])

  // Draw waveform with range selection
  useEffect(() => {
    if (!canvasRef.current || !waveformData) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const data = waveformData.data

    // Clear canvas
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform
    const step = width / data.length
    ctx.strokeStyle = '#4b5563'
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

    // Draw uncertain segment highlight (light red)
    if (currentSegment) {
      const segmentStartSec = timeToSeconds(currentSegment.segment_time)
      const segmentEndSec = segmentStartSec + 2.97  // FRAME_DURATION_SEC from backend
      const totalDuration = waveformData.duration

      const startX = (segmentStartSec / totalDuration) * width
      const endX = (segmentEndSec / totalDuration) * width

      ctx.fillStyle = 'rgba(239, 68, 68, 0.2)'  // Red highlight
      ctx.fillRect(startX, 0, endX - startX, height)
    }

    // Draw selected range (blue overlay)
    if (rangeStart !== null && rangeEnd !== null) {
      const totalDuration = waveformData.duration
      const startX = (rangeStart / totalDuration) * width
      const endX = (rangeEnd / totalDuration) * width

      ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'  // Blue overlay
      ctx.fillRect(startX, 0, endX - startX, height)

      // Draw range markers
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 3

      // Start marker
      ctx.beginPath()
      ctx.moveTo(startX, 0)
      ctx.lineTo(startX, height)
      ctx.stroke()

      // End marker
      ctx.beginPath()
      ctx.moveTo(endX, 0)
      ctx.lineTo(endX, height)
      ctx.stroke()
    }

    // Draw playhead
    if (audioRef.current) {
      const totalDuration = waveformData.duration
      const playheadX = (audioRef.current.currentTime / totalDuration) * width
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, 0)
      ctx.lineTo(playheadX, height)
      ctx.stroke()
    }
  }, [waveformData, rangeStart, rangeEnd, currentSegment, amplitudeScale, zoom])

  // Handle canvas click/drag for range selection
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !waveformData) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const clickTime = (x / rect.width) * waveformData.duration

    setRangeStart(clickTime)
    setRangeEnd(clickTime)
    setIsDragging(true)
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || !canvasRef.current || !waveformData) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const moveTime = (x / rect.width) * waveformData.duration

    setRangeEnd(moveTime)
  }

  const handleCanvasMouseUp = () => {
    setIsDragging(false)

    // Ensure start < end
    if (rangeStart !== null && rangeEnd !== null && rangeStart > rangeEnd) {
      const temp = rangeStart
      setRangeStart(rangeEnd)
      setRangeEnd(temp)
    }
  }

  const handlePlayPause = () => {
    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
    } else {
      // If range is selected, play from range start
      if (rangeStart !== null) {
        audioRef.current.currentTime = rangeStart
      }
      audioRef.current.play()
    }
  }

  // Update playhead and time
  useEffect(() => {
    if (!audioRef.current) return

    const handleTimeUpdate = () => {
      if (!audioRef.current) return
      const currentSeconds = audioRef.current.currentTime
      setCurrentTime(secondsToTime(currentSeconds))

      // Auto-stop at range end
      if (rangeEnd !== null && currentSeconds >= rangeEnd) {
        audioRef.current.pause()
        audioRef.current.currentTime = rangeStart || 0
      }
    }

    audioRef.current.addEventListener('timeupdate', handleTimeUpdate)
    return () => audioRef.current?.removeEventListener('timeupdate', handleTimeUpdate)
  }, [rangeStart, rangeEnd])

  // Redraw playhead during playback (requestAnimationFrame loop)
  useEffect(() => {
    if (!isPlaying || !canvasRef.current || !waveformData || !audioRef.current) return

    let animationId: number

    const drawPlayhead = () => {
      if (!canvasRef.current || !waveformData || !audioRef.current) return

      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const width = canvas.width
      const height = canvas.height
      const data = waveformData.data

      // Clear and redraw everything
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, width, height)

      // Redraw waveform
      const step = width / data.length
      ctx.strokeStyle = '#4b5563'
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

      // Redraw uncertain segment highlight
      if (currentSegment) {
        const segmentStartSec = timeToSeconds(currentSegment.segment_time)
        const segmentEndSec = segmentStartSec + 2.97  // FRAME_DURATION_SEC from backend
        const totalDuration = waveformData.duration

        const startX = (segmentStartSec / totalDuration) * width
        const endX = (segmentEndSec / totalDuration) * width

        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)'
        ctx.fillRect(startX, 0, endX - startX, height)
      }

      // Redraw selected range
      if (rangeStart !== null && rangeEnd !== null) {
        const totalDuration = waveformData.duration
        const startX = (rangeStart / totalDuration) * width
        const endX = (rangeEnd / totalDuration) * width

        ctx.fillStyle = 'rgba(59, 130, 246, 0.3)'
        ctx.fillRect(startX, 0, endX - startX, height)

        ctx.strokeStyle = '#3b82f6'
        ctx.lineWidth = 3

        ctx.beginPath()
        ctx.moveTo(startX, 0)
        ctx.lineTo(startX, height)
        ctx.stroke()

        ctx.beginPath()
        ctx.moveTo(endX, 0)
        ctx.lineTo(endX, height)
        ctx.stroke()
      }

      // Draw playhead
      const totalDuration = waveformData.duration
      const playheadX = (audioRef.current.currentTime / totalDuration) * width
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, 0)
      ctx.lineTo(playheadX, height)
      ctx.stroke()

      animationId = requestAnimationFrame(drawPlayhead)
    }

    drawPlayhead()
    return () => cancelAnimationFrame(animationId)
  }, [isPlaying, waveformData, rangeStart, rangeEnd, currentSegment, amplitudeScale, zoom])

  const handleExport = async (userLabel: string) => {
    if (!currentSegment || rangeStart === null || rangeEnd === null || exporting) return

    setExporting(true)
    try {
      await axios.post('/api/v1/uncertainty/export-range', {
        csv_path: currentSegment.csv_path,
        mp3_path: currentSegment.mp3_path,
        segment_index: currentSegment.segment_index,
        start_time: secondsToTime(rangeStart),
        end_time: secondsToTime(rangeEnd),
        user_label: userLabel
      })

      setSuccessToast({
        show: true,
        message: `✅ Exported ${(rangeEnd - rangeStart).toFixed(1)}s as ${userLabel}`
      })

      // Move to next segment (keep exporting=true until waveform loads)
      setTimeout(() => {
        if (currentIndex < segments.length - 1) {
          setCurrentIndex(prev => prev + 1)
          // setExporting(false) is now handled by loadWaveform's requestAnimationFrame
        } else {
          setSuccessToast({ show: true, message: '🎉 All segments reviewed!' })
          setExporting(false)
        }
      }, 500)  // Reduced delay since we don't need to wait for waveform

    } catch (error: any) {
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Export failed' })
      setExporting(false)
    }
  }

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1)
    }
  }

  const handleUndoExport = async () => {
    if (!currentSegment || exporting) return

    if (!confirm('Undo last export? This will delete the WAV file and remove the tracking entry.')) {
      return
    }

    setExporting(true)
    try {
      const response = await axios.post('/api/v1/uncertainty/undo-export', {
        csv_path: currentSegment.csv_path,
        segment_index: currentSegment.segment_index
      })

      setSuccessToast({
        show: true,
        message: `↩️ Undone export: ${response.data.deleted_wav.split('\\').pop()?.split('/').pop()}`
      })

      // Go back to previous segment
      if (currentIndex > 0) {
        setCurrentIndex(prev => prev - 1)
      }

      setExporting(false)

    } catch (error: any) {
      setErrorToast({
        show: true,
        message: error.response?.data?.detail || 'Undo failed'
      })
      setExporting(false)
    }
  }

  const handleSkip = () => {
    if (currentIndex < segments.length - 1) {
      setCurrentIndex(prev => prev + 1)
    }
  }

  const handleSkipFile = async () => {
    if (!currentSegment || exporting) return

    setExporting(true)
    try {
      // Call backend to mark all remaining uncertain segments from this CSV as skipped
      const response = await axios.post('/api/v1/uncertainty/skip-file', {
        csv_path: currentSegment.csv_path
      })

      const skippedCount = response.data.segments_skipped
      const fileName = currentSegment.mp3_path.split('\\').pop()?.split('/').pop()

      setSuccessToast({
        show: true,
        message: `⏭️ Skipped ${skippedCount} segments from ${fileName}. Reloading...`
      })

      // Reload segments to get fresh list without skipped ones
      setTimeout(() => {
        loadUncertainSegments()
        setCurrentIndex(0)
        setExporting(false)
      }, 1000)

    } catch (error: any) {
      setErrorToast({
        show: true,
        message: error.response?.data?.detail || 'Failed to skip file'
      })
      setExporting(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="text-center py-12">Loading uncertain segments...</div>
      </div>
    )
  }

  if (segments.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-3xl font-bold">🎲 Uncertainty Review</h1>
          <button onClick={onBack} className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
            ← Back to Home
          </button>
        </div>

        {/* Filters - same as main view */}
        <div className="mb-6 bg-white p-4 rounded shadow">
          <h3 className="font-semibold mb-3">Filters</h3>

          {/* Category Filter */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Category</label>
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedCategory(null)}
                className={`px-3 py-1 rounded ${!selectedCategory ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              >
                All
              </button>
              {['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING'].map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1 rounded ${selectedCategory === cat ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Confidence Range Slider */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Confidence Range: {(minConfidence * 100).toFixed(0)}% - {(maxConfidence * 100).toFixed(0)}%
            </label>
            <div className="flex gap-4 items-center">
              <div className="flex-1">
                <label className="text-xs text-gray-600">Min</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={minConfidence * 100}
                  onChange={(e) => setMinConfidence(parseInt(e.target.value) / 100)}
                  className="w-full"
                />
              </div>
              <div className="flex-1">
                <label className="text-xs text-gray-600">Max</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={maxConfidence * 100}
                  onChange={(e) => setMaxConfidence(parseInt(e.target.value) / 100)}
                  className="w-full"
                />
              </div>
            </div>
            <div className="mt-2 flex gap-2">
              <button onClick={() => { setMinConfidence(0); setMaxConfidence(0.7) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
                Uncertain (0-70%)
              </button>
              <button onClick={() => { setMinConfidence(0); setMaxConfidence(1) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
                All (0-100%)
              </button>
              <button onClick={() => { setMinConfidence(0.9); setMaxConfidence(1) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
                High Confidence (90-100%)
              </button>
            </div>
          </div>
        </div>

        <div className="text-center py-12 bg-white rounded shadow">
          <p className="text-xl">🎉 No segments found!</p>
          <p className="text-gray-600 mt-2">
            No segments match current filters (confidence: {(minConfidence * 100).toFixed(0)}-{(maxConfidence * 100).toFixed(0)}%{selectedCategory ? `, category: ${selectedCategory}` : ''}).
          </p>
          <p className="text-gray-600">Try adjusting the filters above or analyze more files.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">🎲 Uncertainty Review</h1>
          {stats && (
            <p className="text-sm text-gray-600 mt-1">
              {stats.remaining} uncertain segments remaining • {stats.total_reviewed} reviewed total
            </p>
          )}
        </div>
        <button onClick={onBack} className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">
          ← Back to Home
        </button>
      </div>

      {/* Filters */}
      <div className="mb-6 bg-white p-4 rounded shadow">
        <h3 className="font-semibold mb-3">Filters</h3>

        {/* Category Filter */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Category</label>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedCategory(null)}
              className={`px-3 py-1 rounded ${!selectedCategory ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            >
              All
            </button>
            {['APPLAUSE', 'MUSIC', 'PUBLIC', 'SPEECH', 'TUNING'].map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1 rounded ${selectedCategory === cat ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Confidence Range Slider */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Confidence Range: {(minConfidence * 100).toFixed(0)}% - {(maxConfidence * 100).toFixed(0)}%
          </label>
          <div className="flex gap-4 items-center">
            <div className="flex-1">
              <label className="text-xs text-gray-600">Min</label>
              <input
                type="range"
                min="0"
                max="100"
                value={minConfidence * 100}
                onChange={(e) => setMinConfidence(parseInt(e.target.value) / 100)}
                className="w-full"
              />
            </div>
            <div className="flex-1">
              <label className="text-xs text-gray-600">Max</label>
              <input
                type="range"
                min="0"
                max="100"
                value={maxConfidence * 100}
                onChange={(e) => setMaxConfidence(parseInt(e.target.value) / 100)}
                className="w-full"
              />
            </div>
          </div>
          <div className="mt-2 flex gap-2">
            <button onClick={() => { setMinConfidence(0); setMaxConfidence(0.7) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
              Uncertain (0-70%)
            </button>
            <button onClick={() => { setMinConfidence(0); setMaxConfidence(1) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
              All (0-100%)
            </button>
            <button onClick={() => { setMinConfidence(0.9); setMaxConfidence(1) }} className="px-2 py-1 text-xs bg-gray-200 rounded">
              High Confidence (90-100%)
            </button>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="mb-6 bg-white p-4 rounded shadow">
        <div className="flex items-center justify-between mb-2">
          <span className="text-lg font-medium">
            Progress: {currentIndex + 1} / {segments.length}
          </span>
          <span className="text-sm text-gray-600">
            {selectedCategory ? `Category: ${selectedCategory}` : 'All categories'}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all"
            style={{ width: `${((currentIndex + 1) / segments.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="bg-white rounded shadow p-6">
        {/* Segment Info */}
        <div className="mb-4">
          <h2 className="text-xl font-semibold">
            📁 {currentSegment.mp3_path.split('\\').pop()?.split('/').pop()}
          </h2>
          <p className="text-gray-600">
            📅 Concert date: <span className="font-medium">
              {(() => {
                // Extract date from path: Y:\...\SORTED\2025\06\27\SONG001.MP3
                const pathParts = currentSegment.mp3_path.split('\\').filter(p => p)
                const year = pathParts[pathParts.length - 4]
                const month = pathParts[pathParts.length - 3]
                const day = pathParts[pathParts.length - 2]
                return `${year}-${month}-${day}`
              })()}
            </span> •
            ⏱️ Segment at {currentSegment.segment_time} •
            Model confidence: <span className="font-bold text-red-600">
              {(currentSegment.confidence * 100).toFixed(0)}%
            </span> ({currentSegment.predicted_class})
          </p>
        </div>

        {/* Waveform with Range Selection */}
        <div className="mb-4">
          <div className="mb-2 text-sm text-gray-700">
            <strong>Drag on waveform to select range</strong> •
            Selected: {rangeStart !== null && rangeEnd !== null ?
              `${secondsToTime(rangeStart)} → ${secondsToTime(rangeEnd)} (${(rangeEnd - rangeStart).toFixed(1)}s)`
              : 'None'}
          </div>
          <div className="flex gap-2 items-center">
            <div ref={scrollContainerRef} className="overflow-x-auto flex-1">
              <canvas
                ref={canvasRef}
                width={1000 * zoom}
                height={150}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseUp}
                className="cursor-crosshair border border-gray-300 rounded"
              />
            </div>
            {/* Amplitude Zoom Slider */}
            <div className="flex flex-col items-center gap-1">
              <span className="text-xs text-gray-500">{amplitudeScale.toFixed(1)}x</span>
              <input
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={amplitudeScale}
                onChange={(e) => setAmplitudeScale(parseFloat(e.target.value))}
                className="h-24 cursor-pointer"
                style={{ writingMode: 'bt-lr', WebkitAppearance: 'slider-vertical' }}
              />
              <span className="text-xs text-gray-500">Zoom</span>
            </div>
          </div>
        </div>

        {/* Audio Controls */}
        <div className="mb-6 flex items-center gap-3">
          <button
            onClick={handlePlayPause}
            className="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play Selected Range'}
          </button>
          <span className="text-sm text-gray-600">
            {currentTime} / {duration}
          </span>
          <span className="text-sm text-gray-500">
            Zoom: {zoom.toFixed(1)}x (scroll to zoom)
          </span>
        </div>

        <audio
          ref={audioRef}
          src={`/api/v1/audio/stream?path=${encodeURIComponent(currentSegment.mp3_path)}`}
          preload="auto"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onLoadedMetadata={(e) => {
            const audio = e.target as HTMLAudioElement
            setDuration(secondsToTime(audio.duration))
          }}
        />

        {/* Loading Overlay */}
        {(loadingWaveform || exporting) && (
          <div className="mb-6 p-6 bg-yellow-50 border-2 border-yellow-300 rounded-lg">
            <p className="text-lg font-bold text-yellow-800">
              {loadingWaveform ? '⏳ Loading waveform...' : '📤 Exporting and loading next segment...'}
            </p>
            <p className="text-sm text-yellow-700 mt-1">
              Please wait before reviewing the next segment
            </p>
          </div>
        )}

        {/* Class Selection Buttons */}
        <div className="mb-6">
          <p className="mb-3 font-medium">What class is the selected range?</p>
          <div className="grid grid-cols-5 gap-3">
            {['MUSIC', 'APPLAUSE', 'SPEECH', 'PUBLIC', 'TUNING'].map(cls => (
              <button
                key={cls}
                onClick={() => handleExport(cls)}
                disabled={rangeStart === null || rangeEnd === null || loadingWaveform || exporting}
                className={`
                  py-4 px-3 rounded-lg font-medium text-lg transition-all
                  ${CLASS_COLORS[cls as keyof typeof CLASS_COLORS].bg}
                  ${CLASS_COLORS[cls as keyof typeof CLASS_COLORS].text}
                  hover:opacity-80
                  ${cls === currentSegment.predicted_class ? 'ring-2 ring-offset-2 ring-blue-500' : ''}
                  disabled:opacity-30 disabled:cursor-not-allowed
                `}
              >
                {exporting ? '...' : cls}
              </button>
            ))}
          </div>
          {(rangeStart === null || rangeEnd === null) && !loadingWaveform && !exporting && (
            <p className="mt-2 text-sm text-red-600">
              ⚠ Select a range on the waveform first
            </p>
          )}
        </div>

        {/* Navigation Buttons */}
        <div className="flex gap-3 justify-between">
          <div className="flex gap-3">
            <button
              onClick={handlePrevious}
              disabled={currentIndex === 0 || loadingWaveform || exporting}
              className="px-6 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>
            <button
              onClick={handleUndoExport}
              disabled={loadingWaveform || exporting}
              className="px-6 py-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-30 disabled:cursor-not-allowed font-medium"
            >
              ↩️ Undo Export
            </button>
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleSkipFile}
              disabled={loadingWaveform || exporting}
              className="px-6 py-2 bg-orange-500 text-white rounded hover:bg-orange-600 disabled:opacity-30 disabled:cursor-not-allowed font-medium"
            >
              ⏭️ Skip This File
            </button>
            <button
              onClick={handleSkip}
              disabled={loadingWaveform || exporting}
              className="px-6 py-2 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Skip Segment →
            </button>
          </div>
        </div>
      </div>

      {/* Toasts */}
      {successToast.show && (
        <Toast
          message={successToast.message}
          type="success"
          onClose={() => setSuccessToast({ show: false, message: '' })}
          duration={2000}
        />
      )}
      {errorToast.show && (
        <Toast
          message={errorToast.message}
          type="error"
          onClose={() => setErrorToast({ show: false, message: '' })}
          duration={3000}
        />
      )}
    </div>
  )
}
