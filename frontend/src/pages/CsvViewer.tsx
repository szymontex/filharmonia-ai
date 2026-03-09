import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import axios from 'axios'
import { CLASS_COLORS } from '../constants/colors'
import StickyPlayer from '../components/StickyPlayer'
import Toast from '../components/Toast'
import { TrackTable } from '../components/TrackTable'
import { CsvSelector } from '../components/CsvSelector'
import { PlayerControls } from '../components/PlayerControls'
import KeyboardHelp from '../components/KeyboardHelp'
import { ImportTracklistDialog } from '../components/ImportTracklistDialog'
import { useExponentialPolling } from '../hooks/useExponentialPolling'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { useTrackEditor } from '../hooks/useTrackEditor'
import { useAutosave } from '../hooks/useAutosave'
import { useUndoRedo } from '../hooks/useUndoRedo'
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts'
import { useConfidenceAdjust } from '../hooks/useConfidenceAdjust'
import { calculateDuration, timeToSeconds } from '../utils/timeCalculations'

// Classification order for number key shortcuts (1-5)
const CLASS_ORDER = ['MUSIC', 'APPLAUSE', 'SPEECH', 'PUBLIC', 'TUNING'] as const


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
  // Track editing state (from hook)
  const {
    tracks,
    setTracks,
    hasUnsavedChanges,
    setHasUnsavedChanges,
    toggleSelect,
    updateName,
    updateStart,
    updateStop,
    updateClass,
    deleteTrack,
    mergeWithNext,
    cutSegmentAtTime,
    addSegmentAtTime,
    addSegmentBelow
  } = useTrackEditor()

  // Audio player state
  const {
    showPlayer,
    togglePlayer,
    closePlayer,
    playingTrackId,
    setPlayingTrackId,
    selectedTrackId,
    setSelectedTrackId,
    seekToTime,
    clearSeekRequest,
    playFromSegment,
    isPlaying: _isPlaying,  // Used indirectly via setIsPlaying callback (updated by StickyPlayer)
    setIsPlaying,
    togglePlaybackRef,
    togglePlayback
  } = useAudioPlayer()

  // Undo/redo state
  const undoRedo = useUndoRedo()

  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
  const [selectedCsv, setSelectedCsv] = useState<string | null>(null)

  // Confidence threshold auto-tuning (learns from user corrections)
  const { threshold: _confidenceThreshold, recordCorrection, refresh: refreshConfidence } = useConfidenceAdjust(selectedCsv)
  const [loading, setLoading] = useState(false)
  const [mp3Path, setMp3Path] = useState<string>('')
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<{ show: boolean; path: string; name: string }>({ show: false, path: '', name: '' })
  const [recordingDate, setRecordingDate] = useState<string | null>(null)
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [exportConfirm, setExportConfirm] = useState<{ show: boolean; count: number }>({ show: false, count: 0 })
  const [exportSummary, setExportSummary] = useState<{ show: boolean; exported: number; skipped: number; errors: number }>({ show: false, exported: 0, skipped: 0, errors: 0 })
  const [analyzingFiles, setAnalyzingFiles] = useState<Map<string, number>>(new Map())  // filename -> progress%
  const [editedCsvs, setEditedCsvs] = useState<Set<string>>(new Set())  // Set of edited CSV paths
  const [exportedSegments, setExportedSegments] = useState<Set<number>>(new Set())  // Set of exported segment indices
  const [csvsWithExports, setCsvsWithExports] = useState<Set<string>>(new Set())  // Set of CSV paths that have exported segments
  const [threshold, setThreshold] = useState(3)  // Threshold for noise filtering
  const [debouncedThreshold, setDebouncedThreshold] = useState(3)  // Debounced threshold for API calls
  const [progressStage, setProgressStage] = useState<string | null>(null)  // Progress indicator for operations
  const abortRef = useRef<AbortController | null>(null)  // AbortController for analysis cancellation
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)  // Keyboard help panel visibility
  const [showImportDialog, setShowImportDialog] = useState(false)  // Import tracklist dialog

  // Fetch function for exponential backoff polling
  const fetchAnalysisStatus = useCallback(async () => {
    const response = await axios.get('/api/v1/analyze/batch')
    const runningJobs = response.data.filter((job: any) => job.status === 'running')

    const newAnalyzingFiles = new Map<string, number>()
    for (const job of runningJobs) {
      const detailRes = await axios.get(`/api/v1/analyze/batch/${job.job_id}`)
      if (detailRes.data.current_file) {
        newAnalyzingFiles.set(detailRes.data.current_file, detailRes.data.current_file_progress || 0)
      }
    }

    return { runningJobs, analyzingFiles: newAnalyzingFiles }
  }, [])

  // Exponential backoff polling for analysis jobs
  const { data: analysisData, startPolling, stopPolling } = useExponentialPolling(
    fetchAnalysisStatus,
    {
      initialInterval: 1000,   // Start at 1 second
      maxInterval: 10000,      // Max 10 seconds
      multiplier: 1.5,         // 1s -> 1.5s -> 2.25s -> 3.4s -> 5s -> 7.5s -> 10s
      resetOnChange: true      // Reset to fast when status changes
    }
  )

  // Update analyzingFiles when polling data changes
  useEffect(() => {
    if (analysisData?.analyzingFiles) {
      setAnalyzingFiles(analysisData.analyzingFiles)
    }
  }, [analysisData])

  // Load initial data and start polling on mount
  useEffect(() => {
    loadCsvList()
    loadEditedList()
    loadCsvsWithExports()
    startPolling()
    return () => {
      stopPolling()
      abortRef.current?.abort()  // Cancel any in-flight analysis on unmount
    }
  }, [startPolling, stopPolling])

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

  // Refresh confidence threshold when recording changes
  useEffect(() => {
    refreshConfidence()
  }, [selectedCsv, refreshConfidence])

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
      loadCsv(selectedCsv, { skipAutosaveCheck: true })
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

  const loadCsv = async (csvPath: string, { skipAutosaveCheck = false } = {}) => {
    // Abort any in-flight request and create new controller
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setProgressStage('Loading...')
    setSelectedCsv(csvPath)

    try {
      let pathToLoad = csvPath

      // Only check autosave on initial file load, not on threshold changes
      if (!skipAutosaveCheck) {
        const autosaveCheck = await axios.get(`/api/v1/csv/check-autosave?path=${encodeURIComponent(csvPath)}`, {
          signal: abortRef.current.signal
        })

        if (autosaveCheck.data.has_autosave && autosaveCheck.data.autosave_newer) {
          const useAutosave = window.confirm(
            `Found newer autosave from ${new Date(autosaveCheck.data.autosave_time).toLocaleString()}.\n\nLoad autosave instead of original?`
          )
          if (useAutosave) {
            pathToLoad = autosaveCheck.data.autosave_path
          }
        }
      }

      // Load and parse CSV with threshold
      const res = await axios.get(`/api/v1/csv/parse?path=${encodeURIComponent(pathToLoad)}&threshold=${debouncedThreshold}`, {
        signal: abortRef.current.signal
      })
      setTracks(res.data.tracks)
      undoRedo.resetHistory(res.data.tracks)  // Reset undo/redo history on file switch
      setHasUnsavedChanges(false)

      // Load exported segments for this CSV
      await loadExportedSegments(csvPath)

      // Resolve MP3 path from CSV via backend API
      try {
        const response = await axios.get(`/api/v1/files/mp3-for-csv?csv_path=${encodeURIComponent(csvPath)}`, {
          signal: abortRef.current.signal
        })
        setMp3Path(response.data.mp3_path)
        setRecordingDate(response.data.recording_date)
      } catch (error) {
        // Silently ignore cancellation errors
        if (axios.isCancel(error)) {
          return
        }
        console.error('Error resolving MP3 path:', error)
      }
    } catch (error) {
      // Silently ignore cancellation errors
      if (axios.isCancel(error)) {
        return
      }
      // Let axios interceptor handle other errors
      throw error
    } finally {
      setLoading(false)
      setProgressStage(null)
    }
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

  const saveToFile = async () => {
    if (!selectedCsv) return

    try {
      setProgressStage('Saving...')
      await axios.post('/api/v1/csv/save', {
        path: selectedCsv,
        tracks: tracks
      })
      // Clean up autosave file after successful save
      await axios.delete(`/api/v1/csv/discard-autosave?path=${encodeURIComponent(selectedCsv)}`).catch(() => {})
      setHasUnsavedChanges(false)
      setShowSaveModal(true)
      setTimeout(() => setShowSaveModal(false), 2000)

      // Mark as edited and refresh list
      setEditedCsvs(prev => new Set(prev).add(selectedCsv))
      loadEditedList()
    } catch (error: any) {
      console.error('Save failed:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to save CSV' })
    } finally {
      setProgressStage(null)
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
    setDeleteConfirm({ show: true, path: csvPath, name: csvPath.split(/[/\\]/).pop() || '' })
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

  // Autosave when tracks change
  const { lastSave: lastAutosave } = useAutosave({
    data: tracks,
    enabled: hasUnsavedChanges && selectedCsv !== null,
    saveFn: async (trackData) => {
      await axios.post('/api/v1/csv/autosave', {
        path: selectedCsv,
        tracks: trackData
      })
    }
  })

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
      const [, month, day] = recordingDate.split('-')
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

  const handleImportTracklist = (updatedTracks: typeof tracks) => {
    undoRedo.pushState(tracks)
    setTracks(updatedTracks)
    setHasUnsavedChanges(true)
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
      const segments = selected.map((track) => {
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
      }, {
        signal: abortRef.current?.signal
      })

      const summary = response.data.summary
      setExportSummary({
        show: true,
        exported: summary.exported,
        skipped: summary.skipped,
        errors: summary.errors
      })

      // Refresh exported segments list
      if (selectedCsv) {
        await loadExportedSegments(selectedCsv)
      }
      await loadCsvsWithExports()

    } catch (error: any) {
      // Silently ignore cancellation errors
      if (axios.isCancel(error)) {
        return
      }
      console.error('Export error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Export failed' })
    }
  }

  // Wrapper functions for track mutations with undo/redo
  const wrappedToggleSelect = useCallback((id: string) => {
    undoRedo.pushState(tracks)
    toggleSelect(id)
  }, [tracks, undoRedo, toggleSelect])

  const wrappedUpdateName = useCallback((id: string, name: string) => {
    updateName(id, name)
  }, [updateName])

  const pushUndoForName = useCallback(() => {
    undoRedo.pushState(tracks)
  }, [tracks, undoRedo])

  const wrappedUpdateClass = useCallback((id: string, predicted_class: string) => {
    undoRedo.pushState(tracks)
    updateClass(id, predicted_class)
  }, [tracks, undoRedo, updateClass])

  const wrappedUpdateStart = useCallback((id: string, start: string) => {
    undoRedo.pushState(tracks)
    updateStart(id, start)
  }, [tracks, undoRedo, updateStart])

  const wrappedUpdateStop = useCallback((id: string, stop: string) => {
    undoRedo.pushState(tracks)
    updateStop(id, stop)
  }, [tracks, undoRedo, updateStop])

  const wrappedDeleteTrack = useCallback((id: string) => {
    undoRedo.pushState(tracks)
    deleteTrack(id)
    recordCorrection('delete')
  }, [tracks, undoRedo, deleteTrack, recordCorrection])

  const wrappedMergeWithNext = useCallback((id: string) => {
    undoRedo.pushState(tracks)
    mergeWithNext(id)
  }, [tracks, undoRedo, mergeWithNext])

  const wrappedAddSegmentBelow = useCallback((id: string) => {
    undoRedo.pushState(tracks)
    addSegmentBelow(id)
    recordCorrection('add')
  }, [tracks, undoRedo, addSegmentBelow, recordCorrection])

  // Undo/redo handlers - apply returned tracks directly
  const handleUndo = useCallback(() => {
    const restored = undoRedo.undo()
    if (restored) {
      setTracks(restored)
      setHasUnsavedChanges(true)
    }
  }, [undoRedo, setTracks, setHasUnsavedChanges])

  const handleRedo = useCallback(() => {
    const restored = undoRedo.redo()
    if (restored) {
      setTracks(restored)
      setHasUnsavedChanges(true)
    }
  }, [undoRedo, setTracks, setHasUnsavedChanges])

  // Keyboard shortcut handlers
  const keyboardHandlers = useMemo(() => ({
    'space': () => {
      if (showPlayer) {
        togglePlayback()
      } else {
        togglePlayer()
      }
    },
    'ctrl+s': () => {
      saveToFile()
    },
    'ctrl+z': handleUndo,
    'ctrl+shift+z': handleRedo,
    'ctrl+y': handleRedo,
    '1': () => {
      if (selectedTrackId) {
        wrappedUpdateClass(selectedTrackId, CLASS_ORDER[0])
      }
    },
    '2': () => {
      if (selectedTrackId) {
        wrappedUpdateClass(selectedTrackId, CLASS_ORDER[1])
      }
    },
    '3': () => {
      if (selectedTrackId) {
        wrappedUpdateClass(selectedTrackId, CLASS_ORDER[2])
      }
    },
    '4': () => {
      if (selectedTrackId) {
        wrappedUpdateClass(selectedTrackId, CLASS_ORDER[3])
      }
    },
    '5': () => {
      if (selectedTrackId) {
        wrappedUpdateClass(selectedTrackId, CLASS_ORDER[4])
      }
    },
    'shift+?': () => {
      setShowKeyboardHelp(prev => !prev)
    },
    'escape': () => {
      if (showKeyboardHelp) {
        setShowKeyboardHelp(false)
      }
    }
  }), [showPlayer, togglePlayer, togglePlayback, saveToFile, handleUndo, handleRedo, selectedTrackId, wrappedUpdateClass, showKeyboardHelp])

  useKeyboardShortcuts(keyboardHandlers)

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
        <CsvSelector
          files={csvFiles}
          selectedCsv={selectedCsv}
          onSelect={loadCsv}
          onDelete={deleteCsv}
          analyzingFiles={analyzingFiles}
          editedCsvs={editedCsvs}
          csvsWithExports={csvsWithExports}
        />

        {/* Tracks Table */}
        {loading ? (
          <div className="text-center py-12">Loading...</div>
        ) : tracks.length > 0 ? (
          <div className="bg-white rounded-lg shadow" style={{ marginBottom: showPlayer ? '280px' : '0' }}>
            <div className="p-4 border-b flex justify-between items-center">
              <div>
                <div className="text-lg font-semibold">
                  {mp3Path.split(/[/\\]/).pop()?.replace('.MP3', '').replace('.mp3', '')}
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
              <PlayerControls
                showPlayer={showPlayer}
                mp3Path={mp3Path}
                onTogglePlayer={togglePlayer}
                hasUnsavedChanges={hasUnsavedChanges}
                onSave={saveToFile}
                onDiscard={discardChanges}
                threshold={threshold}
                onThresholdChange={setThreshold}
                thresholdDisabled={hasUnsavedChanges}
                selectedCount={tracks.filter(t => t.selected).length}
                onExportToTraining={exportToTrainingData}
                onCopyTracklist={copyTracklistToClipboard}
                onImportTracklist={() => setShowImportDialog(true)}
                progressStage={progressStage}
                onUndo={handleUndo}
                onRedo={handleRedo}
                canUndo={undoRedo.canUndo}
                canRedo={undoRedo.canRedo}
                onShowKeyboardHelp={() => setShowKeyboardHelp(true)}
              />
            </div>

            <TrackTable
              tracks={tracks}
              exportedSegments={exportedSegments}
              playingTrackId={playingTrackId}
              selectedTrackId={selectedTrackId}
              onToggleSelect={wrappedToggleSelect}
              onUpdateName={wrappedUpdateName}
              onNameFocus={pushUndoForName}
              onUpdateClass={wrappedUpdateClass}
              onUpdateStart={wrappedUpdateStart}
              onUpdateStop={wrappedUpdateStop}
              onDelete={wrappedDeleteTrack}
              onMergeWithNext={wrappedMergeWithNext}
              onAddSegmentBelow={wrappedAddSegmentBelow}
              onSelectTrack={setSelectedTrackId}
              onPlayFrom={playFromSegment}
              onUndoExport={handleUndoExport}
              onSelectAll={(selected) => {
                setTracks(tracks.map(t => ({ ...t, selected })))
                setHasUnsavedChanges(true)
              }}
            />
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
          onClose={closePlayer}
          onTrackUpdate={handleTrackUpdate}
          onBoundaryUpdate={handleBoundaryUpdate}
          selectedTrackId={selectedTrackId}
          onTrackSelect={setSelectedTrackId}
          seekToTime={seekToTime}
          onSeekComplete={clearSeekRequest}
          recordingDate={recordingDate}
          onAddSegment={addSegmentAtTime}
          onCutSegment={cutSegmentAtTime}
          onPlayingTrackChange={setPlayingTrackId}
          onTogglePlaybackRef={togglePlaybackRef}
          onPlayingStateChange={setIsPlaying}
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

      {/* Import Tracklist Dialog */}
      <ImportTracklistDialog
        show={showImportDialog}
        onClose={() => setShowImportDialog(false)}
        csvName={selectedCsv}
        tracks={tracks}
        onImport={handleImportTracklist}
      />

      {/* Keyboard Help Panel */}
      <KeyboardHelp
        show={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
      />
    </div>
  )
}
