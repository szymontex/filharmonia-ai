import { useState, useCallback } from 'react'
import { calculateDuration, parseTimeToSeconds, secondsToTimeFormat } from '../utils/timeCalculations'

export interface Track {
  id: string
  selected: boolean
  name: string
  predicted_class: string
  start: string
  stop: string
  duration: string
}

export interface UseTrackEditorReturn {
  tracks: Track[]
  setTracks: React.Dispatch<React.SetStateAction<Track[]>>
  hasUnsavedChanges: boolean
  setHasUnsavedChanges: React.Dispatch<React.SetStateAction<boolean>>

  // Operations
  toggleSelect: (id: string) => void
  updateName: (id: string, name: string) => void
  updateStart: (id: string, start: string) => void
  updateStop: (id: string, stop: string) => void
  updateClass: (id: string, predicted_class: string) => void
  deleteTrack: (id: string) => void
  mergeWithNext: (id: string) => void
  cutSegmentAtTime: (timeStr: string) => void
  addSegmentAtTime: (timeStr: string, totalDuration?: number) => void
  addSegmentBelow: (id: string) => void
}

export function useTrackEditor(): UseTrackEditorReturn {
  const [tracks, setTracks] = useState<Track[]>([])
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

  const toggleSelect = useCallback((id: string) => {
    setTracks(prev => prev.map(t =>
      t.id === id ? { ...t, selected: !t.selected } : t
    ))
    setHasUnsavedChanges(true)
  }, [])

  const updateName = useCallback((id: string, name: string) => {
    setTracks(prev => prev.map(t =>
      t.id === id ? { ...t, name } : t
    ))
    setHasUnsavedChanges(true)
  }, [])

  const updateStart = useCallback((id: string, start: string) => {
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
  }, [])

  const updateStop = useCallback((id: string, stop: string) => {
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
  }, [])

  const updateClass = useCallback((id: string, predicted_class: string) => {
    setTracks(prev => prev.map(t =>
      t.id === id ? { ...t, predicted_class } : t
    ))
    setHasUnsavedChanges(true)
  }, [])

  const deleteTrack = useCallback((id: string) => {
    setTracks(prevTracks => {
      const idx = prevTracks.findIndex(t => t.id === id)
      if (idx === -1) return prevTracks

      // Merge with previous track if it exists
      if (idx > 0) {
        const prevTrack = prevTracks[idx - 1]
        const deletedTrack = prevTracks[idx]

        const updatedPrevTrack = {
          ...prevTrack,
          stop: deletedTrack.stop,
          duration: calculateDuration(prevTrack.start, deletedTrack.stop)
        }

        return [
          ...prevTracks.slice(0, idx - 1),
          updatedPrevTrack,
          ...prevTracks.slice(idx + 1)
        ]
      } else if (idx === 0 && prevTracks.length > 1) {
        // If deleting first track, extend next track to cover its time
        const nextTrack = prevTracks[1]
        const deletedTrack = prevTracks[0]

        const updatedNextTrack = {
          ...nextTrack,
          start: deletedTrack.start,
          duration: calculateDuration(deletedTrack.start, nextTrack.stop)
        }

        return [
          updatedNextTrack,
          ...prevTracks.slice(2)
        ]
      } else {
        // Only one track - just remove it
        return prevTracks.filter(t => t.id !== id)
      }
    })
    setHasUnsavedChanges(true)
  }, [])

  const mergeWithNext = useCallback((id: string) => {
    setTracks(prevTracks => {
      const idx = prevTracks.findIndex(t => t.id === id)
      if (idx === -1 || idx === prevTracks.length - 1) return prevTracks

      const current = prevTracks[idx]
      const next = prevTracks[idx + 1]

      const merged = {
        ...current,
        stop: next.stop,
        duration: calculateDuration(current.start, next.stop)
      }

      return [
        ...prevTracks.slice(0, idx),
        merged,
        ...prevTracks.slice(idx + 2)
      ]
    })
    setHasUnsavedChanges(true)
  }, [])

  const cutSegmentAtTime = useCallback((timeStr: string) => {
    const timeSeconds = parseTimeToSeconds(timeStr)

    setTracks(prevTracks => {
      // Find which segment this time falls into
      let segmentIndex = -1
      for (let i = 0; i < prevTracks.length; i++) {
        const startSeconds = parseTimeToSeconds(prevTracks[i].start)
        const stopSeconds = parseTimeToSeconds(prevTracks[i].stop)

        if (timeSeconds > startSeconds && timeSeconds < stopSeconds) {
          segmentIndex = i
          break
        }
      }

      if (segmentIndex === -1) {
        return prevTracks
      }

      const currentSegment = prevTracks[segmentIndex]
      const cutTime = secondsToTimeFormat(timeSeconds)

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

      return [
        ...prevTracks.slice(0, segmentIndex),
        firstSegment,
        secondSegment,
        ...prevTracks.slice(segmentIndex + 1)
      ]
    })
    setHasUnsavedChanges(true)
  }, [])

  const addSegmentAtTime = useCallback((timeStr: string, totalDuration?: number) => {
    const timeSeconds = parseTimeToSeconds(timeStr)

    setTracks(prevTracks => {
      // Find which segment this time falls into
      let segmentIndex = -1
      for (let i = 0; i < prevTracks.length; i++) {
        const startSeconds = parseTimeToSeconds(prevTracks[i].start)
        const stopSeconds = parseTimeToSeconds(prevTracks[i].stop)

        if (timeSeconds >= startSeconds && timeSeconds < stopSeconds) {
          segmentIndex = i
          break
        }
      }

      if (segmentIndex === -1) {
        return prevTracks
      }

      const currentSegment = prevTracks[segmentIndex]
      const currentStopSeconds = parseTimeToSeconds(currentSegment.stop)

      // Create new segment: 8 seconds duration
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

      return [
        ...prevTracks.slice(0, segmentIndex),
        updatedCurrentSegment,
        newSegment,
        nextSegment,
        ...prevTracks.slice(segmentIndex + 1)
      ]
    })
    setHasUnsavedChanges(true)
  }, [])

  const addSegmentBelow = useCallback((id: string) => {
    setTracks(prevTracks => {
      const idx = prevTracks.findIndex(t => t.id === id)
      if (idx === -1) return prevTracks

      const current = prevTracks[idx]
      const stopSeconds = parseTimeToSeconds(current.stop)

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
        predicted_class: current.predicted_class,
        start: newStart,
        stop: newStop,
        duration: calculateDuration(newStart, newStop)
      }

      const updatedTracks = [...prevTracks]
      // Update next segment's start time if it exists
      if (idx + 1 < prevTracks.length) {
        const nextSegment = { ...updatedTracks[idx + 1], start: newStop }
        nextSegment.duration = calculateDuration(nextSegment.start, nextSegment.stop)
        updatedTracks[idx + 1] = nextSegment
      }

      // Insert right after current segment
      return [
        ...updatedTracks.slice(0, idx + 1),
        newSegment,
        ...updatedTracks.slice(idx + 1)
      ]
    })
    setHasUnsavedChanges(true)
  }, [])

  return {
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
  }
}
