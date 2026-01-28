import { useState, useEffect, useRef, useCallback } from 'react'

export interface UseAutosaveOptions<T> {
  /** Data to autosave */
  data: T
  /** Whether autosave is enabled (e.g., hasUnsavedChanges) */
  enabled: boolean
  /** Function to perform the save */
  saveFn: (data: T) => Promise<void>
  /** Delay before saving (0 = immediate, default 0) */
  delay?: number
  /** Callback on successful save */
  onSave?: () => void
  /** Callback on save error */
  onError?: (error: Error) => void
}

export interface UseAutosaveReturn {
  /** Time of last successful save */
  lastSave: Date | null
  /** Whether a save is currently in progress */
  isSaving: boolean
  /** Manually trigger a save */
  manualSave: () => Promise<void>
}

/**
 * Hook for automatic saving of data when it changes
 *
 * @example
 * const { lastSave, isSaving } = useAutosave({
 *   data: tracks,
 *   enabled: hasUnsavedChanges && selectedCsv !== null,
 *   saveFn: async (data) => {
 *     await axios.post('/api/v1/csv/autosave', { path: selectedCsv, tracks: data })
 *   }
 * })
 */
export function useAutosave<T>({
  data,
  enabled,
  saveFn,
  delay = 0,
  onSave,
  onError
}: UseAutosaveOptions<T>): UseAutosaveReturn {
  const [lastSave, setLastSave] = useState<Date | null>(null)
  const [isSaving, setIsSaving] = useState(false)

  // Refs to avoid stale closures in async callbacks
  const dataRef = useRef(data)
  const mountedRef = useRef(true)

  // Keep data ref in sync
  useEffect(() => {
    dataRef.current = data
  }, [data])

  // Track mounted state for cleanup
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  // Autosave effect
  useEffect(() => {
    if (!enabled) return

    const performSave = async () => {
      if (!mountedRef.current) return

      setIsSaving(true)
      try {
        await saveFn(dataRef.current)
        if (mountedRef.current) {
          setLastSave(new Date())
          onSave?.()
          console.log('Autosaved at', new Date().toLocaleTimeString())
        }
      } catch (error) {
        console.error('Autosave failed:', error)
        if (mountedRef.current) {
          onError?.(error as Error)
        }
      } finally {
        if (mountedRef.current) {
          setIsSaving(false)
        }
      }
    }

    if (delay > 0) {
      // Debounced save
      const timer = setTimeout(performSave, delay)
      return () => clearTimeout(timer)
    } else {
      // Immediate save
      performSave()
    }
  }, [data, enabled, delay, saveFn, onSave, onError])

  // Manual save function
  const manualSave = useCallback(async () => {
    setIsSaving(true)
    try {
      await saveFn(dataRef.current)
      setLastSave(new Date())
      onSave?.()
    } catch (error) {
      onError?.(error as Error)
      throw error
    } finally {
      setIsSaving(false)
    }
  }, [saveFn, onSave, onError])

  return {
    lastSave,
    isSaving,
    manualSave
  }
}
