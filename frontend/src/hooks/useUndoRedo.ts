import { useState, useCallback, useRef } from 'react'
import type { Track } from './useTrackEditor'

export interface UndoRedoReturn {
  undo: () => Track[] | null   // returns tracks to apply, or null if nothing to undo
  redo: () => Track[] | null   // returns tracks to apply, or null if nothing to redo
  canUndo: boolean
  canRedo: boolean
  pushState: (tracks: Track[]) => void   // called before each mutation
  resetHistory: (tracks: Track[]) => void // called on file switch
  present: Track[]  // current state after undo/redo
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
 * undo()/redo() now return the new tracks directly so the caller
 * can apply them immediately, avoiding race conditions with useEffect.
 */
export function useUndoRedo(): UndoRedoReturn {
  const historyRef = useRef<HistoryState>({
    past: [],
    present: [],
    future: []
  })
  // version counter to trigger re-renders when history changes
  const [, setVersion] = useState(0)
  const bump = () => setVersion(v => v + 1)

  const pushState = useCallback((tracks: Track[]) => {
    const prev = historyRef.current
    historyRef.current = {
      past: [...prev.past, prev.present].slice(-19),
      present: tracks,
      future: []
    }
    bump()
  }, [])

  const undo = useCallback((): Track[] | null => {
    const prev = historyRef.current
    if (prev.past.length === 0) return null

    const previous = prev.past[prev.past.length - 1]
    historyRef.current = {
      past: prev.past.slice(0, -1),
      present: previous,
      future: [prev.present, ...prev.future]
    }
    bump()
    return previous
  }, [])

  const redo = useCallback((): Track[] | null => {
    const prev = historyRef.current
    if (prev.future.length === 0) return null

    const next = prev.future[0]
    historyRef.current = {
      past: [...prev.past, prev.present],
      present: next,
      future: prev.future.slice(1)
    }
    bump()
    return next
  }, [])

  const resetHistory = useCallback((tracks: Track[]) => {
    historyRef.current = {
      past: [],
      present: tracks,
      future: []
    }
    bump()
  }, [])

  const h = historyRef.current
  return {
    undo,
    redo,
    canUndo: h.past.length > 0,
    canRedo: h.future.length > 0,
    pushState,
    resetHistory,
    present: h.present
  }
}
