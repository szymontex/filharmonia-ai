import { useState, useCallback } from 'react'
import type { Track } from './useTrackEditor'

export interface UndoRedoReturn {
  undo: () => void
  redo: () => void
  canUndo: boolean
  canRedo: boolean
  pushState: (tracks: Track[]) => void   // called before each mutation
  resetHistory: (tracks: Track[]) => void // called on file switch
}

interface HistoryState {
  past: Track[][]
  present: Track[]
  future: Track[][]
}

/**
 * Snapshot-based undo/redo hook for Track arrays
 *
 * Maintains up to 20 previous states. Future clears on new edit.
 * History persists across saves, clears on file switch.
 *
 * @example
 * const { undo, redo, canUndo, canRedo, pushState, resetHistory } = useUndoRedo()
 *
 * // Before mutation
 * pushState(tracks)
 *
 * // On file switch
 * resetHistory(newTracks)
 */
export function useUndoRedo(): UndoRedoReturn {
  const [history, setHistory] = useState<HistoryState>({
    past: [],
    present: [],
    future: []
  })

  const pushState = useCallback((tracks: Track[]) => {
    setHistory(prev => ({
      past: [...prev.past, prev.present].slice(-19), // Keep max 20 (current becomes 20th)
      present: tracks,
      future: [] // Clear future on new edit
    }))
  }, [])

  const undo = useCallback(() => {
    setHistory(prev => {
      if (prev.past.length === 0) return prev // No-op if past empty

      const previous = prev.past[prev.past.length - 1]
      const newPast = prev.past.slice(0, -1)

      return {
        past: newPast,
        present: previous,
        future: [prev.present, ...prev.future]
      }
    })
  }, [])

  const redo = useCallback(() => {
    setHistory(prev => {
      if (prev.future.length === 0) return prev // No-op if future empty

      const next = prev.future[0]
      const newFuture = prev.future.slice(1)

      return {
        past: [...prev.past, prev.present],
        present: next,
        future: newFuture
      }
    })
  }, [])

  const resetHistory = useCallback((tracks: Track[]) => {
    setHistory({
      past: [],
      present: tracks,
      future: []
    })
  }, [])

  return {
    undo,
    redo,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    pushState,
    resetHistory
  }
}
