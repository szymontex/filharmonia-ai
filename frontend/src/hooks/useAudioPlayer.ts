import { useState, useCallback } from 'react'

export interface UseAudioPlayerReturn {
  /** Whether the player panel is visible */
  showPlayer: boolean
  /** Toggle player visibility */
  togglePlayer: () => void
  /** Show player */
  openPlayer: () => void
  /** Hide player */
  closePlayer: () => void

  /** ID of track currently being played */
  playingTrackId: string | null
  /** Set the playing track ID */
  setPlayingTrackId: (id: string | null) => void

  /** ID of track currently hovered/selected in table */
  selectedTrackId: string | null
  /** Set the selected track ID */
  setSelectedTrackId: (id: string | null) => void

  /** Time to seek to (HH:MM:SS format) */
  seekToTime: string | null
  /** Clear seek request after player handles it */
  clearSeekRequest: () => void

  /** Play from a specific segment - shows player and seeks to start time */
  playFromSegment: (startTime: string, trackId: string) => void
}

/**
 * Hook for managing audio player state
 *
 * @example
 * const {
 *   showPlayer,
 *   togglePlayer,
 *   playFromSegment,
 *   playingTrackId,
 *   ...
 * } = useAudioPlayer()
 */
export function useAudioPlayer(): UseAudioPlayerReturn {
  const [showPlayer, setShowPlayer] = useState(false)
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null)
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)
  const [seekToTime, setSeekToTime] = useState<string | null>(null)

  const togglePlayer = useCallback(() => {
    setShowPlayer(prev => !prev)
  }, [])

  const openPlayer = useCallback(() => {
    setShowPlayer(true)
  }, [])

  const closePlayer = useCallback(() => {
    setShowPlayer(false)
  }, [])

  const clearSeekRequest = useCallback(() => {
    setSeekToTime(null)
  }, [])

  const playFromSegment = useCallback((startTime: string, trackId: string) => {
    // Show player if not already visible
    setShowPlayer(true)
    // Trigger seek to this time
    setSeekToTime(startTime)
    // Mark this track as playing
    setPlayingTrackId(trackId)
  }, [])

  return {
    showPlayer,
    togglePlayer,
    openPlayer,
    closePlayer,
    playingTrackId,
    setPlayingTrackId,
    selectedTrackId,
    setSelectedTrackId,
    seekToTime,
    clearSeekRequest,
    playFromSegment
  }
}
