/**
 * Per-recording confidence threshold auto-tuning.
 *
 * Learns from user corrections:
 * - User deletes a predicted segment (false positive) -> raise threshold
 * - User adds a new segment (false negative) -> lower threshold
 *
 * Persists to localStorage so thresholds survive page refreshes.
 */

const STORAGE_KEY = 'filharmonia_thresholds'
const DEFAULT_THRESHOLD = 0.7
const LEARNING_RATE = 0.05
const MIN_THRESHOLD = 0.3
const MAX_THRESHOLD = 0.95

export interface RecordingThreshold {
  recordingId: string
  threshold: number
  falsePositives: number
  falseNegatives: number
  corrections: number
}

export class ConfidenceThresholdLearner {
  private thresholds: Map<string, RecordingThreshold> = new Map()

  constructor() {
    this.load()
  }

  getThreshold(recordingId: string): number {
    return this.thresholds.get(recordingId)?.threshold ?? DEFAULT_THRESHOLD
  }

  getRecord(recordingId: string): RecordingThreshold | undefined {
    return this.thresholds.get(recordingId)
  }

  recordCorrection(recordingId: string, type: 'delete' | 'add'): number {
    let record = this.thresholds.get(recordingId)
    if (!record) {
      record = {
        recordingId,
        threshold: DEFAULT_THRESHOLD,
        falsePositives: 0,
        falseNegatives: 0,
        corrections: 0,
      }
    }

    record.corrections++

    if (type === 'delete') {
      // User removed a prediction = false positive -> raise threshold
      record.falsePositives++
      record.threshold = Math.min(MAX_THRESHOLD, record.threshold + LEARNING_RATE)
    } else {
      // User added a label = false negative -> lower threshold
      record.falseNegatives++
      record.threshold = Math.max(MIN_THRESHOLD, record.threshold - LEARNING_RATE)
    }

    this.thresholds.set(recordingId, record)
    this.persist()

    return record.threshold
  }

  persist(): void {
    try {
      const data = Object.fromEntries(this.thresholds)
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
    } catch {
      console.warn('Failed to persist confidence thresholds to localStorage')
    }
  }

  load(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const data = JSON.parse(raw) as Record<string, RecordingThreshold>
        this.thresholds = new Map(Object.entries(data))
      }
    } catch {
      console.warn('Failed to load confidence thresholds from localStorage')
    }
  }
}

/** Singleton instance shared across the app */
export const thresholdLearner = new ConfidenceThresholdLearner()
