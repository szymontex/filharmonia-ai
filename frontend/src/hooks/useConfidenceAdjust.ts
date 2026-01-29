import { useState, useCallback } from 'react'
import { thresholdLearner } from '../utils/confidenceThreshold'

/**
 * React hook for per-recording confidence threshold auto-tuning.
 *
 * Returns the current threshold for the given recording and a function
 * to record user corrections (which adjusts the threshold).
 */
export function useConfidenceAdjust(recordingId: string | null) {
  const [threshold, setThreshold] = useState(() =>
    recordingId ? thresholdLearner.getThreshold(recordingId) : 0.7
  )

  const recordCorrection = useCallback(
    (type: 'delete' | 'add') => {
      if (!recordingId) return
      const newThreshold = thresholdLearner.recordCorrection(recordingId, type)
      setThreshold(newThreshold)
    },
    [recordingId]
  )

  // Re-sync when recordingId changes
  const refresh = useCallback(() => {
    if (recordingId) {
      setThreshold(thresholdLearner.getThreshold(recordingId))
    }
  }, [recordingId])

  return { threshold, recordCorrection, refresh }
}
