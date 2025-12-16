import { useState, useEffect } from 'react'
import axios from 'axios'
import { CLASS_COLORS } from '../constants/colors'
import StickyPlayer from '../components/StickyPlayer'
import Toast from '../components/Toast'

interface Track {
  id: string
  selected: boolean
  name: string
  predicted_class: string
  start: string
  stop: string
  duration: string
}

interface CsvFile {
  path: string
  name: string
  date: string
}

interface CsvViewerProps {
  onBack?: () => void
  initialCsv?: string | null
}

export default function CsvViewer({ onBack, initialCsv }: CsvViewerProps = {}) {
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
  const [selectedCsv, setSelectedCsv] = useState<string | null>(null)
  const [tracks, setTracks] = useState<Track[]>([])
  const [loading, setLoading] = useState(false)
  const [mp3Path, setMp3Path] = useState<string>('')
  const [showPlayer, setShowPlayer] = useState(false)
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)
  const [seekToTime, setSeekToTime] = useState<string | null>(null)
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [lastAutosave, setLastAutosave] = useState<Date | null>(null)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<{ show: boolean; path: string; name: string }>({ show: false, path: '', name: '' })
  const [recordingDate, setRecordingDate] = useState<string | null>(null)
  const [showExportModal, setShowExportModal] = useState(false)
  const [exportedCount, setExportedCount] = useState(0)
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [exportConfirm, setExportConfirm] = useState<{ show: boolean; count: number }>({ show: false, count: 0 })
  const [exportSummary, setExportSummary] = useState<{ show: boolean; exported: number; skipped: number; errors: number }>({ show: false, exported: 0, skipped: 0, errors: 0 })
  const [analyzingFiles, setAnalyzingFiles] = useState<Map<string, number>>(new Map())  // filename -> progress%
  const [editedCsvs, setEditedCsvs] = useState<Set<string>>(new Set())  // Set of edited CSV paths
  const [exportedSegments, setExportedSegments] = useState<Set<number>>(new Set())  // Set of exported segment indices
  const [csvsWithExports, setCsvsWithExports] = useState<Set<string>>(new Set())  // Set of CSV paths that have exported segments
  const [threshold, setThreshold] = useState(5)  // Threshold for noise filtering
  const [debouncedThreshold, setDebouncedThreshold] = useState(5)  // Debounced threshold for API calls

  useEffect(() => {
    loadCsvList()
    loadEditedList()
    loadCsvsWithExports()

    // Poll for active analysis jobs every 2 seconds
    const interval = setInterval(async () => {
      try {
        const response = await axios.get('/api/v1/analyze/batch')
        const runningJobs = response.data.filter((job: any) => job.status === 'running')

        const newAnalyzingFiles = new Map<string, number>()
        for (const job of runningJobs) {
          // Fetch detailed status to get current file
          const detailRes = await axios.get(`/api/v1/analyze/batch/${job.job_id}`)
          if (detailRes.data.current_file) {
            newAnalyzingFiles.set(detailRes.data.current_file, detailRes.data.current_file_progress || 0)
          }
        }

        setAnalyzingFiles(newAnalyzingFiles)
      } catch (error) {
        console.error('Error fetching analysis status:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const loadEditedList = async () => {
    try {
      const res = await axios.get('/api/v1/csv/edited-list')
      setEditedCsvs(new Set(res.data.edited_files))
    } catch (error) {
      console.error('Error loading edited CSV list:', error)
    }
  }

  const loadExportedSegments = async (csvPath: string) => {
    try {
      const res = await axios.get(`/api/v1/export/check-exported?csv_path=${encodeURIComponent(csvPath)}`)
      setExportedSegments(new Set(res.data.exported_indices))

      // Update csvsWithExports if this CSV has any exports
      if (res.data.exported_indices.length > 0) {
        setCsvsWithExports(prev => new Set(prev).add(csvPath))
      }
    } catch (error) {
      console.error('Error loading exported segments:', error)
    }
  }

  const handleUndoExport = async (segmentIndex: number) => {
    if (!selectedCsv) return

    try {
      await axios.delete(`/api/v1/export/segment?csv_path=${encodeURIComponent(selectedCsv)}&segment_index=${segmentIndex}`)

      // Remove from exported segments set
      setExportedSegments(prev => {
        const next = new Set(prev)
        next.delete(segmentIndex)
        return next
      })

      setSuccessToast({ show: true, message: 'Export undone successfully' })
    } catch (error: any) {
      console.error('Error undoing export:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to undo export' })
    }
  }

  const loadCsvsWithExports = async () => {
    try {
      // Read exported_segments.csv to find all CSVs with exports
      const response = await axios.get('/api/v1/export/all-exported-csvs')
      setCsvsWithExports(new Set(response.data.csv_paths))
    } catch (error) {
      console.error('Error loading CSVs with exports:', error)
    }
  }

  useEffect(() => {
    if (initialCsv) {
      loadCsv(initialCsv)
    }
  }, [initialCsv])

  // Debounce threshold changes (wait 500ms after user stops sliding)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedThreshold(threshold)
    }, 500)

    return () => clearTimeout(timer)
  }, [threshold])

  // Reload tracks when debounced threshold changes
  useEffect(() => {
    if (selectedCsv && !hasUnsavedChanges) {
      loadCsv(selectedCsv)
    }
  }, [debouncedThreshold])

  const loadCsvList = async () => {
    try {
      const res = await axios.get('/api/v1/files/analysis-results')
      console.log('CSV files loaded:', res.data)
      setCsvFiles(res.data)
    } catch (error) {
      console.error('Error loading CSV files:', error)
    }
  }

  const loadCsv = async (csvPath: string) => {
    setLoading(true)
    setSelectedCsv(csvPath)

    // Check for autosave
    const autosaveCheck = await axios.get(`/api/v1/csv/check-autosave?path=${encodeURIComponent(csvPath)}`)

    let pathToLoad = csvPath

    if (autosaveCheck.data.has_autosave && autosaveCheck.data.autosave_newer) {
      const useAutosave = window.confirm(
        `Found newer autosave from ${new Date(autosaveCheck.data.autosave_time).toLocaleString()}.\n\nLoad autosave instead of original?`
      )
      if (useAutosave) {
        pathToLoad = autosaveCheck.data.autosave_path
      }
    }

    // Load and parse CSV with threshold
    const res = await axios.get(`/api/v1/csv/parse?path=${encodeURIComponent(pathToLoad)}&threshold=${debouncedThreshold}`)
    setTracks(res.data.tracks)
    setHasUnsavedChanges(false)

    // Load exported segments for this CSV
    await loadExportedSegments(csvPath)

    setLoading(false)

    // Extract MP3 path from CSV path (remove _autosave if present)
    // predictions_SONG059_2025-05-13.csv or predictions_SONG059_2025-05-13_14-30.csv or predictions_SONG059_2025-05-13_autosave.csv
    // -> find SONG059.MP3 in SORTED/2025/05/13/
    const cleanPath = csvPath.replace('_autosave', '')
    // Match: predictions_{songName}_{YYYY-MM-DD}[_{HH-MM}].csv
    const match = cleanPath.match(/predictions_(.+?)_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv/)
    if (match) {
      const [, songName, year, month, day] = match
      const mp3 = `Y:\\!_FILHARMONIA\\SORTED\\${year}\\${month}\\${day}\\${songName}.MP3`
      setMp3Path(mp3)
      setRecordingDate(`${year}-${month}-${day}`)
    }
  }

  const togglePlayer = () => {
    setShowPlayer(!showPlayer)
  }

  const toggleSelect = (id: string) => {
    setTracks(tracks.map(t =>
      t.id === id ? { ...t, selected: !t.selected } : t
    ))
    setHasUnsavedChanges(true)
  }

  const updateName = (id: string, name: string) => {
    setTracks(tracks.map(t =>
      t.id === id ? { ...t, name } : t
    ))
    setHasUnsavedChanges(true)
  }

  const updateStart = (id: string, start: string) => {
    setTracks(prevTracks => {
      const trackIndex = prevTracks.findIndex(t => t.id === id)
      if (trackIndex === -1) return prevTracks

      const updatedTracks = [...prevTracks]
      const currentTrack = { ...updatedTracks[trackIndex], start }

      if (currentTrack.start && currentTrack.stop) {
        currentTrack.duration = calculateDuration(currentTrack.start, currentTrack.stop)
      }

      updatedTracks[trackIndex] = currentTrack

      // Update stop time of previous track to match this track's start time
      if (trackIndex > 0) {
        const prevTrack = { ...updatedTracks[trackIndex - 1], stop: start }
        if (prevTrack.start && prevTrack.stop) {
          prevTrack.duration = calculateDuration(prevTrack.start, prevTrack.stop)
        }
        updatedTracks[trackIndex - 1] = prevTrack
      }

      return updatedTracks
    })
    setHasUnsavedChanges(true)
  }

  const updateStop = (id: string, stop: string) => {
    setTracks(prevTracks => {
      const trackIndex = prevTracks.findIndex(t => t.id === id)
      if (trackIndex === -1) return prevTracks

      const updatedTracks = [...prevTracks]
      const currentTrack = { ...updatedTracks[trackIndex], stop }

      if (currentTrack.start && currentTrack.stop) {
        currentTrack.duration = calculateDuration(currentTrack.start, currentTrack.stop)
      }

      updatedTracks[trackIndex] = currentTrack

      // Update start time of next track to match this track's stop time
      if (trackIndex < updatedTracks.length - 1) {
        const nextTrack = { ...updatedTracks[trackIndex + 1], start: stop }
        if (nextTrack.start && nextTrack.stop) {
          nextTrack.duration = calculateDuration(nextTrack.start, nextTrack.stop)
        }
        updatedTracks[trackIndex + 1] = nextTrack
      }

      return updatedTracks
    })
    setHasUnsavedChanges(true)
  }

  const updateClass = (id: string, predicted_class: string) => {
    setTracks(tracks.map(t =>
      t.id === id ? { ...t, predicted_class } : t
    ))
    setHasUnsavedChanges(true)
  }

  const deleteTrack = (id: string) => {
    const idx = tracks.findIndex(t => t.id === id)
    if (idx === -1) return

    // Merge with previous track if it exists
    if (idx > 0) {
      const prevTrack = tracks[idx - 1]
      const deletedTrack = tracks[idx]

      // Extend previous track to cover deleted track's time
      const updatedPrevTrack = {
        ...prevTrack,
        stop: deletedTrack.stop,
        duration: calculateDuration(prevTrack.start, deletedTrack.stop)
      }

      const newTracks = [
        ...tracks.slice(0, idx - 1),
        updatedPrevTrack,
        ...tracks.slice(idx + 1)
      ]
      setTracks(newTracks)
    } else if (idx === 0 && tracks.length > 1) {
      // If deleting first track, extend next track to cover its time
      const nextTrack = tracks[1]
      const deletedTrack = tracks[0]

      const updatedNextTrack = {
        ...nextTrack,
        start: deletedTrack.start,
        duration: calculateDuration(deletedTrack.start, nextTrack.stop)
      }

      const newTracks = [
        updatedNextTrack,
        ...tracks.slice(2)
      ]
      setTracks(newTracks)
    } else {
      // Only one track - just remove it
      setTracks(tracks.filter(t => t.id !== id))
    }

    setHasUnsavedChanges(true)
  }

  const cutSegmentAtTime = (timeStr: string) => {
    const timeSeconds = parseInt(timeStr.split(':')[0]) * 3600 + parseInt(timeStr.split(':')[1]) * 60 + parseInt(timeStr.split(':')[2])

    // Find which segment this time falls into
    let segmentIndex = -1
    for (let i = 0; i < tracks.length; i++) {
      const startSeconds = parseInt(tracks[i].start.split(':')[0]) * 3600 + parseInt(tracks[i].start.split(':')[1]) * 60 + parseInt(tracks[i].start.split(':')[2])
      const stopSeconds = parseInt(tracks[i].stop.split(':')[0]) * 3600 + parseInt(tracks[i].stop.split(':')[1]) * 60 + parseInt(tracks[i].stop.split(':')[2])

      if (timeSeconds > startSeconds && timeSeconds < stopSeconds) {
        segmentIndex = i
        break
      }
    }

    if (segmentIndex === -1) {
      // Time is outside existing segments or exactly at boundary
      return
    }

    const currentSegment = tracks[segmentIndex]
    const cutTime = secondsToTimeFormat(timeSeconds)

    // Split into two segments with same class
    const firstSegment: Track = {
      ...currentSegment,
      id: `track-${Date.now()}-first`,
      stop: cutTime,
      duration: calculateDuration(currentSegment.start, cutTime)
    }

    const secondSegment: Track = {
      ...currentSegment,
      id: `track-${Date.now()}-second`,
      start: cutTime,
      duration: calculateDuration(cutTime, currentSegment.stop)
    }

    // Replace current segment with two segments
    const newTracks = [
      ...tracks.slice(0, segmentIndex),
      firstSegment,
      secondSegment,
      ...tracks.slice(segmentIndex + 1)
    ]

    setTracks(newTracks)
    setHasUnsavedChanges(true)
  }

  const addSegmentAtTime = (timeStr: string, totalDuration?: number) => {
    const timeSeconds = parseInt(timeStr.split(':')[0]) * 3600 + parseInt(timeStr.split(':')[1]) * 60 + parseInt(timeStr.split(':')[2])

    // Find which segment this time falls into
    let segmentIndex = -1
    for (let i = 0; i < tracks.length; i++) {
      const startSeconds = parseInt(tracks[i].start.split(':')[0]) * 3600 + parseInt(tracks[i].start.split(':')[1]) * 60 + parseInt(tracks[i].start.split(':')[2])
      const stopSeconds = parseInt(tracks[i].stop.split(':')[0]) * 3600 + parseInt(tracks[i].stop.split(':')[1]) * 60 + parseInt(tracks[i].stop.split(':')[2])

      if (timeSeconds >= startSeconds && timeSeconds < stopSeconds) {
        segmentIndex = i
        break
      }
    }

    if (segmentIndex === -1) {
      // Time is outside existing segments, don't add
      return
    }

    const currentSegment = tracks[segmentIndex]
    const currentStopSeconds = parseInt(currentSegment.stop.split(':')[0]) * 3600 + parseInt(currentSegment.stop.split(':')[1]) * 60 + parseInt(currentSegment.stop.split(':')[2])

    // Create new segment: 6-10 seconds duration (let's use 8 seconds as default)
    const duration = 8
    const newStartSeconds = timeSeconds
    const newStopSeconds = Math.min(timeSeconds + duration, currentStopSeconds)

    const newStart = secondsToTimeFormat(newStartSeconds)
    const newStop = secondsToTimeFormat(newStopSeconds)

    const newSegment: Track = {
      id: `track-${Date.now()}`,
      selected: false,
      name: '',
      predicted_class: 'MUSIC',
      start: newStart,
      stop: newStop,
      duration: calculateDuration(newStart, newStop)
    }

    // Update current segment's stop time to newStart
    const updatedCurrentSegment = {
      ...currentSegment,
      stop: newStart,
      duration: calculateDuration(currentSegment.start, newStart)
    }

    // Create next segment from newStop to original stop
    const nextSegment: Track = {
      id: `track-${Date.now()}-next`,
      selected: false,
      name: currentSegment.name,
      predicted_class: currentSegment.predicted_class,
      start: newStop,
      stop: currentSegment.stop,
      duration: calculateDuration(newStop, currentSegment.stop)
    }

    // Replace current segment with: updated current + new segment + next segment
    const newTracks = [
      ...tracks.slice(0, segmentIndex),
      updatedCurrentSegment,
      newSegment,
      nextSegment,
      ...tracks.slice(segmentIndex + 1)
    ]

    setTracks(newTracks)
    setHasUnsavedChanges(true)
  }

  const secondsToTimeFormat = (seconds: number): string => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const mergeWithNext = (id: string) => {
    const idx = tracks.findIndex(t => t.id === id)
    if (idx === -1 || idx === tracks.length - 1) return

    const current = tracks[idx]
    const next = tracks[idx + 1]

    const merged = {
      ...current,
      stop: next.stop,
      duration: calculateDuration(current.start, next.stop)
    }

    setTracks([
      ...tracks.slice(0, idx),
      merged,
      ...tracks.slice(idx + 2)
    ])
    setHasUnsavedChanges(true)
  }

  const addSegmentBelow = (id: string) => {
    const idx = tracks.findIndex(t => t.id === id)
    if (idx === -1) return

    const current = tracks[idx]
    const stopSeconds = parseInt(current.stop.split(':')[0]) * 3600 + parseInt(current.stop.split(':')[1]) * 60 + parseInt(current.stop.split(':')[2])

    // Create new segment starting at current segment's stop time, 8 seconds duration
    const duration = 8
    const newStartSeconds = stopSeconds
    const newStopSeconds = stopSeconds + duration

    const newStart = secondsToTimeFormat(newStartSeconds)
    const newStop = secondsToTimeFormat(newStopSeconds)

    const newSegment: Track = {
      id: `track-${Date.now()}`,
      selected: false,
      name: '',
      predicted_class: current.predicted_class, // Copy class from segment above
      start: newStart,
      stop: newStop,
      duration: calculateDuration(newStart, newStop)
    }

    // Update next segment's start time if it exists
    const updatedTracks = [...tracks]
    if (idx + 1 < tracks.length) {
      const nextSegment = { ...updatedTracks[idx + 1], start: newStop }
      nextSegment.duration = calculateDuration(nextSegment.start, nextSegment.stop)
      updatedTracks[idx + 1] = nextSegment
    }

    // Insert right after current segment
    const newTracks = [
      ...updatedTracks.slice(0, idx + 1),
      newSegment,
      ...updatedTracks.slice(idx + 1)
    ]

    setTracks(newTracks)
    setHasUnsavedChanges(true)
  }

  const calculateDuration = (start: string, stop: string): string => {
    const [sh, sm, ss] = start.split(':').map(Number)
    const [eh, em, es] = stop.split(':').map(Number)

    const startSec = sh * 3600 + sm * 60 + ss
    const endSec = eh * 3600 + em * 60 + es
    const diffSec = endSec - startSec

    const minutes = Math.floor(diffSec / 60)
    const seconds = diffSec % 60

    return `${minutes}'${seconds}"`
  }

  const handleTrackUpdate = (trackId: string, updates: { start?: string; stop?: string }) => {
    setTracks(prevTracks => prevTracks.map(t => {
      if (t.id !== trackId) return t

      const newTrack = { ...t, ...updates }

      // Recalculate duration whenever start or stop changes
      if (newTrack.start && newTrack.stop) {
        newTrack.duration = calculateDuration(newTrack.start, newTrack.stop)
      }

      return newTrack
    }))
    setHasUnsavedChanges(true)
  }

  const handleBoundaryUpdate = (prevTrackId: string, nextTrackId: string, time: string) => {
    setTracks(prevTracks => prevTracks.map(t => {
      if (t.id === prevTrackId) {
        // Update stop time of previous segment
        const newTrack = { ...t, stop: time }
        if (newTrack.start && newTrack.stop) {
          newTrack.duration = calculateDuration(newTrack.start, newTrack.stop)
        }
        return newTrack
      } else if (t.id === nextTrackId) {
        // Update start time of next segment
        const newTrack = { ...t, start: time }
        if (newTrack.start && newTrack.stop) {
          newTrack.duration = calculateDuration(newTrack.start, newTrack.stop)
        }
        return newTrack
      }
      return t
    }))
    setHasUnsavedChanges(true)
  }

  const playFromSegment = (startTime: string, trackId: string) => {
    // Show player if not already visible
    if (!showPlayer) {
      setShowPlayer(true)
    }
    // Trigger seek to this time
    setSeekToTime(startTime)
    // Mark this track as playing
    setPlayingTrackId(trackId)
  }

  const saveToFile = async () => {
    if (!selectedCsv) return

    try {
      await axios.post('/api/v1/csv/save', {
        path: selectedCsv,
        tracks: tracks
      })
      setHasUnsavedChanges(false)
      setShowSaveModal(true)
      setTimeout(() => setShowSaveModal(false), 2000)

      // Mark as edited and refresh list
      setEditedCsvs(prev => new Set(prev).add(selectedCsv))
      loadEditedList()
    } catch (error: any) {
      console.error('Save failed:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to save CSV' })
    }
  }

  const discardChanges = async () => {
    if (!selectedCsv) return

    const confirm = window.confirm('Discard all unsaved changes and reload original?')
    if (!confirm) return

    try {
      await axios.delete(`/api/v1/csv/discard-autosave?path=${encodeURIComponent(selectedCsv)}`)
      await loadCsv(selectedCsv)
    } catch (error) {
      console.error('Discard failed:', error)
    }
  }

  const deleteCsv = (csvPath: string, event: React.MouseEvent) => {
    event.stopPropagation()
    setDeleteConfirm({ show: true, path: csvPath, name: csvPath.split('\\').pop() || '' })
  }

  const confirmDelete = async () => {
    try {
      await axios.delete(`/api/v1/files/delete-csv?path=${encodeURIComponent(deleteConfirm.path)}`)
      await loadCsvList()
      if (selectedCsv === deleteConfirm.path) {
        setSelectedCsv(null)
        setTracks([])
      }
      setDeleteConfirm({ show: false, path: '', name: '' })
    } catch (error: any) {
      console.error('Delete failed:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to delete CSV' })
    }
  }

  // Handle Enter key for delete confirmation
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (deleteConfirm.show && e.key === 'Enter') {
        confirmDelete()
      } else if (deleteConfirm.show && e.key === 'Escape') {
        setDeleteConfirm({ show: false, path: '', name: '' })
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [deleteConfirm])

  // Autosave immediately after tracks change
  useEffect(() => {
    if (!selectedCsv || !hasUnsavedChanges) return

    const performAutosave = async () => {
      try {
        await axios.post('/api/v1/csv/autosave', {
          path: selectedCsv,
          tracks: tracks
        })
        setLastAutosave(new Date())
        console.log('Autosaved at', new Date().toLocaleTimeString())
      } catch (error) {
        console.error('Autosave failed:', error)
      }
    }

    performAutosave()
  }, [tracks, selectedCsv])

  const exportSelected = () => {
    const selected = tracks.filter(t => t.selected)
    const output = selected.map((t, i) =>
      `${i + 1}. ${t.name || 'Untitled'} (${t.duration})`
    ).join('\n')

    navigator.clipboard.writeText(output)
    setExportedCount(selected.length)
    setShowExportModal(true)
    setTimeout(() => setShowExportModal(false), 2000)
  }

  const copyTracklistToClipboard = () => {
    // Filter MUSIC segments only
    const musicTracks = tracks.filter(t => t.predicted_class === 'MUSIC')

    if (musicTracks.length === 0) {
      setErrorToast({ show: true, message: 'No MUSIC segments found' })
      return
    }

    // Format: date on top, then each track: title and duration in M'S" format
    let output = ''

    // Add date at the top (extract from recordingDate state: YYYY-MM-DD)
    if (recordingDate) {
      const [year, month, day] = recordingDate.split('-')
      output += `${day}.${month}\n`
    }

    // Add each MUSIC track
    musicTracks.forEach(track => {
      if (track.name && track.name.trim()) {
        output += `${track.name}\n`
        output += `${track.duration}\n\n`
      }
    })

    // Copy to clipboard
    navigator.clipboard.writeText(output.trim())
      .then(() => {
        setSuccessToast({ show: true, message: `Copied ${musicTracks.length} MUSIC track${musicTracks.length !== 1 ? 's' : ''} to clipboard` })
      })
      .catch(err => {
        console.error('Copy failed:', err)
        setErrorToast({ show: true, message: 'Failed to copy to clipboard' })
      })
  }

  const exportToTrainingData = () => {
    if (!selectedCsv || !mp3Path) {
      setErrorToast({ show: true, message: 'No CSV or MP3 selected' })
      return
    }

    const selected = tracks.filter(t => t.selected)
    if (selected.length === 0) {
      setErrorToast({ show: true, message: 'No segments selected for export' })
      return
    }

    // Show confirmation toast
    setExportConfirm({ show: true, count: selected.length })
  }

  const performExport = async () => {
    setExportConfirm({ show: false, count: 0 })

    const selected = tracks.filter(t => t.selected)

    try {
      // Convert tracks to segments with indices
      const segments = selected.map((track, idx) => {
        const trackIndex = tracks.findIndex(t => t.id === track.id)
        return {
          start: timeToSeconds(track.start),
          stop: timeToSeconds(track.stop),
          predicted_class: track.predicted_class,
          segment_index: trackIndex,
          segment_time: track.start
        }
      })

      const response = await axios.post('/api/v1/export/training-data', {
        csv_path: selectedCsv,
        mp3_path: mp3Path,
        segments: segments
      })

      const summary = response.data.summary
      setExportSummary({
        show: true,
        exported: summary.exported,
        skipped: summary.skipped,
        errors: summary.errors
      })

      // Refresh exported segments list
      await loadExportedSegments(selectedCsv)
      await loadCsvsWithExports()

    } catch (error: any) {
      console.error('Export error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Export failed' })
    }
  }

  const timeToSeconds = (timeStr: string): number => {
    const parts = timeStr.split(':').map(Number)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]
  }

  const getClassColor = (cls: string) => {
    const config = CLASS_COLORS[cls as keyof typeof CLASS_COLORS]
    if (!config) return 'bg-gray-100 text-gray-800'
    return `${config.bg} ${config.text}`
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {onBack && (
          <button
            onClick={onBack}
            className="text-blue-600 hover:text-blue-800 mb-2"
          >
            ← Back to Home
          </button>
        )}
        <h1 className="text-3xl font-bold mb-6">CSV Track Editor</h1>

        {/* Legend */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Category Legend</h2>
          <div className="flex gap-3">
            {Object.entries(CLASS_COLORS).map(([name, config]) => (
              <span key={name} className={`px-3 py-1 rounded text-sm font-medium ${config.bg} ${config.text}`}>
                {name}
              </span>
            ))}
          </div>
        </div>

        {/* CSV File Selector */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Select Analysis Result</h2>
          <div className="grid grid-cols-1 gap-3 max-h-60 overflow-y-auto">
            {csvFiles.map(file => {
              // Extract song name from predictions_SONG042_2025-09-27.csv or predictions_SONG042_2025-09-27_14-30.csv
              const songMatch = file.name.match(/predictions_(.+?)_\d{4}-\d{2}-\d{2}/)
              const songName = songMatch ? songMatch[1] : file.name

              // Extract time from filename (if present): predictions_SONG042_2025-09-27_14-30.csv
              const timeMatch = file.name.match(/_(\d{2})-(\d{2})\.csv$/)
              const timeStr = timeMatch ? `${timeMatch[1]}:${timeMatch[2]}` : null

              // Format date as DD.MM.YYYY
              const dateParts = file.date.split('-')
              const formattedDate = dateParts.length === 3
                ? `${dateParts[2]}.${dateParts[1]}.${dateParts[0]}`
                : file.date

              // Check if this file is currently being analyzed
              const isAnalyzing = analyzingFiles.has(songName + '.MP3') || analyzingFiles.has(songName + '.mp3')
              const analyzingProgress = (analyzingFiles.get(songName + '.MP3') || analyzingFiles.get(songName + '.mp3')) ?? 0

              return (
                <div
                  key={file.path}
                  className={`relative p-4 rounded border hover:bg-blue-50 cursor-pointer ${
                    selectedCsv === file.path ? 'bg-blue-100 border-blue-500' : 'border-gray-200'
                  }`}
                  onClick={() => loadCsv(file.path)}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div className="text-xl font-bold text-blue-600">{formattedDate}</div>
                    {timeStr && (
                      <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-sm font-semibold rounded">
                        {timeStr}
                      </span>
                    )}
                    {editedCsvs.has(file.path) && (
                      <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs font-semibold rounded">
                        ✏️ EDITED
                      </span>
                    )}
                    {csvsWithExports.has(file.path) && (
                      <span className="px-2 py-0.5 bg-purple-100 text-purple-800 text-xs font-semibold rounded">
                        📦 EXPORTED
                      </span>
                    )}
                    {isAnalyzing && (
                      <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded animate-pulse">
                        ⏳ Analyzing {analyzingProgress.toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <div className="text-sm font-medium text-gray-800">{songName}</div>
                  <div className="text-xs text-gray-500 mt-1">{file.name}</div>
                  <button
                    onClick={(e) => deleteCsv(file.path, e)}
                    className="absolute top-3 right-3 text-red-500 hover:text-red-700 hover:bg-red-100 rounded px-2 py-1 text-lg"
                    title="Delete CSV"
                  >
                    ×
                  </button>
                </div>
              )
            })}
          </div>
        </div>

        {/* Tracks Table */}
        {loading ? (
          <div className="text-center py-12">Loading...</div>
        ) : tracks.length > 0 ? (
          <div className="bg-white rounded-lg shadow" style={{ marginBottom: showPlayer ? '280px' : '0' }}>
            <div className="p-4 border-b flex justify-between items-center">
              <div>
                <div className="text-lg font-semibold">
                  {mp3Path.split('\\').pop()?.replace('.MP3', '').replace('.mp3', '')}
                  <span className="mx-2 text-gray-400">•</span>
                  <span className="text-blue-600">
                    {selectedCsv ? (() => {
                      const cleanPath = selectedCsv.replace('_autosave', '')
                      const match = cleanPath.match(/predictions_.+?_(\d{4})-(\d{2})-(\d{2})(?:_\d{2}-\d{2})?\.csv/)
                      if (match) {
                        const [, year, month, day] = match
                        return `${day}.${month}.${year}`
                      }
                      return ''
                    })() : ''}
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  {tracks.length} tracks ({tracks.filter(t => t.selected).length} selected)
                </div>
                {hasUnsavedChanges && (
                  <div className="text-sm text-orange-600">
                    Unsaved changes {lastAutosave && `• Last autosave: ${lastAutosave.toLocaleTimeString()}`}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-4">
                {/* Threshold Slider */}
                <div className="flex items-center gap-3 mr-4 bg-gray-50 px-4 py-2 rounded-lg border border-gray-200">
                  <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
                    Noise Filter:
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    value={threshold}
                    onChange={(e) => setThreshold(parseInt(e.target.value))}
                    className="w-32"
                    disabled={hasUnsavedChanges}
                  />
                  <span className="text-sm font-semibold text-blue-600 min-w-[4rem]">
                    {threshold} segs
                  </span>
                </div>
                <button
                  onClick={togglePlayer}
                  disabled={!mp3Path}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-300"
                >
                  {showPlayer ? '🔇 Hide Player' : '🎵 Show Player'}
                </button>
                <button
                  onClick={saveToFile}
                  disabled={!hasUnsavedChanges}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-300"
                >
                  💾 Save
                </button>
                <button
                  onClick={discardChanges}
                  disabled={!hasUnsavedChanges}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-300"
                >
                  🗑️ Discard
                </button>
                <button
                  onClick={exportToTrainingData}
                  disabled={tracks.filter(t => t.selected).length === 0}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-300"
                  title="Export selected segments to TRAINING DATA folder"
                >
                  📦 Export to Training
                </button>

                <button
                  onClick={copyTracklistToClipboard}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                  title="Copy tracklist to clipboard (MUSIC segments only)"
                >
                  📋 Copy Tracklist
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={tracks.length > 0 && tracks.every(t => t.selected)}
                          onChange={(e) => {
                            const newSelected = e.target.checked
                            setTracks(tracks.map(t => ({ ...t, selected: newSelected })))
                            setHasUnsavedChanges(true)
                          }}
                          className="w-4 h-4 cursor-pointer"
                        />
                        <span>Select</span>
                      </div>
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Start</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Stop</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Play</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {tracks.map((track, idx) => {
                    const isExported = exportedSegments.has(idx)
                    const isPlaying = playingTrackId === track.id
                    const isHovered = selectedTrackId === track.id

                    let bgColor = ''
                    let hoverClass = 'hover:bg-blue-100'

                    if (isPlaying) {
                      bgColor = 'bg-green-100 border-l-4 border-green-500'
                      hoverClass = 'hover:bg-green-200'  // Lighter green on hover when playing
                    } else if (isHovered) {
                      bgColor = 'bg-blue-100'
                    } else if (isExported) {
                      bgColor = 'bg-purple-50'
                    }

                    return (
                    <tr
                      key={track.id}
                      onMouseEnter={() => setSelectedTrackId(track.id)}
                      onMouseLeave={() => setSelectedTrackId(null)}
                      className={`cursor-pointer transition-colors ${bgColor} ${hoverClass}`}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => toggleSelect(track.id)}
                            className="text-xl"
                          >
                            {track.selected ? '✓' : '✗'}
                          </button>
                          {isExported && (
                            <div className="flex items-center gap-1">
                              <span className="px-1.5 py-0.5 bg-purple-100 text-purple-800 text-xs font-semibold rounded" title="Already exported to training data">
                                📦
                              </span>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  handleUndoExport(idx)
                                }}
                                className="px-1.5 py-0.5 bg-red-100 text-red-700 text-xs font-semibold rounded hover:bg-red-200"
                                title="Undo export - delete from training data"
                              >
                                ↩
                              </button>
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="text"
                          value={track.name}
                          onChange={(e) => updateName(track.id, e.target.value)}
                          className="w-full px-2 py-1 border rounded"
                          placeholder="Track name..."
                        />
                      </td>
                      <td className="px-4 py-3">
                        <select
                          value={track.predicted_class}
                          onChange={(e) => updateClass(track.id, e.target.value)}
                          className={`px-2 py-1 rounded text-sm font-medium border ${getClassColor(track.predicted_class)}`}
                        >
                          {Object.keys(CLASS_COLORS).map(cls => (
                            <option key={cls} value={cls}>{cls}</option>
                          ))}
                        </select>
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="text"
                          value={track.start}
                          onChange={(e) => updateStart(track.id, e.target.value)}
                          className="w-20 px-2 py-1 border rounded text-sm"
                          placeholder="HH:MM:SS"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <input
                          type="text"
                          value={track.stop}
                          onChange={(e) => updateStop(track.id, e.target.value)}
                          className="w-20 px-2 py-1 border rounded text-sm"
                          placeholder="HH:MM:SS"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => playFromSegment(track.start, track.id)}
                          className="px-2 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                        >
                          ▶
                        </button>
                      </td>
                      <td className="px-4 py-3 text-sm font-medium">{track.duration}</td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <button
                            onClick={() => addSegmentBelow(track.id)}
                            className="px-2 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                            title="Add new segment below this one"
                          >
                            + Below
                          </button>
                          <button
                            onClick={() => mergeWithNext(track.id)}
                            className="px-2 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                          >
                            Merge ↓
                          </button>
                          <button
                            onClick={() => deleteTrack(track.id)}
                            className="px-2 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  )})}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            Select a CSV file to view tracks
          </div>
        )}
      </div>

      {/* Sticky Player */}
      {showPlayer && mp3Path && (
        <StickyPlayer
          mp3Path={mp3Path}
          tracks={tracks}
          onClose={() => setShowPlayer(false)}
          onTrackUpdate={handleTrackUpdate}
          onBoundaryUpdate={handleBoundaryUpdate}
          selectedTrackId={selectedTrackId}
          onTrackSelect={setSelectedTrackId}
          seekToTime={seekToTime}
          onSeekComplete={() => setSeekToTime(null)}
          recordingDate={recordingDate}
          onAddSegment={addSegmentAtTime}
          onCutSegment={cutSegmentAtTime}
          onPlayingTrackChange={setPlayingTrackId}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirm.show && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-sm mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete CSV?</h3>
            <p className="text-sm text-gray-600 mb-4">
              Are you sure you want to delete <strong>{deleteConfirm.name}</strong>?
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setDeleteConfirm({ show: false, path: '', name: '' })}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
              >
                Cancel (Esc)
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Delete (Enter)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Save Success Toast */}
      <Toast
        show={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        title="Saved Successfully!"
        message="Your changes have been saved."
        icon="✅"
        color="green"
        index={0}
        autoClose={3000}
      />

      {/* Success Toast (generic) */}
      <Toast
        show={successToast.show}
        onClose={() => setSuccessToast({ show: false, message: '' })}
        title="Success!"
        message={successToast.message}
        icon="✅"
        color="green"
        index={showSaveModal ? 1 : 0}
        autoClose={3000}
      />

      {/* Error Toast */}
      <Toast
        show={errorToast.show}
        onClose={() => setErrorToast({ show: false, message: '' })}
        title="Error"
        message={errorToast.message}
        icon="❌"
        color="red"
        index={(showSaveModal ? 1 : 0) + (successToast.show ? 1 : 0)}
        autoClose={5000}
      />

      {/* Export Confirmation Toast */}
      <Toast
        show={exportConfirm.show}
        onClose={() => setExportConfirm({ show: false, count: 0 })}
        title="Export to Training Data?"
        message={`Export ${exportConfirm.count} segment${exportConfirm.count !== 1 ? 's' : ''}? Already exported segments will be skipped.`}
        icon="📦"
        color="blue"
        index={(showSaveModal ? 1 : 0) + (successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0)}
        autoClose={0}
        actions={[
          {
            label: 'Export',
            onClick: performExport,
            color: 'primary'
          },
          {
            label: 'Cancel',
            onClick: () => setExportConfirm({ show: false, count: 0 }),
            color: 'secondary'
          }
        ]}
      />

      {/* Export Summary Toast */}
      <Toast
        show={exportSummary.show}
        onClose={() => setExportSummary({ show: false, exported: 0, skipped: 0, errors: 0 })}
        title="Export Complete!"
        message={`Exported: ${exportSummary.exported} • Skipped: ${exportSummary.skipped} • Errors: ${exportSummary.errors}`}
        icon="✅"
        color="purple"
        index={(showSaveModal ? 1 : 0) + (successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0) + (exportConfirm.show ? 1 : 0)}
        autoClose={6000}
      />
    </div>
  )
}
