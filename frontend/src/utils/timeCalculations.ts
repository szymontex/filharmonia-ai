/**
 * Time calculation utilities for audio segment editing
 */

/**
 * Calculate duration between two HH:MM:SS timestamps
 * @returns Duration in M'S" format (e.g., "3'45\"")
 */
export function calculateDuration(start: string, stop: string): string {
  const [sh, sm, ss] = start.split(':').map(Number)
  const [eh, em, es] = stop.split(':').map(Number)

  const startSec = sh * 3600 + sm * 60 + ss
  const endSec = eh * 3600 + em * 60 + es
  const diffSec = endSec - startSec

  const minutes = Math.floor(diffSec / 60)
  const seconds = diffSec % 60

  return `${minutes}'${seconds}"`
}

/**
 * Convert HH:MM:SS timestamp to total seconds
 */
export function timeToSeconds(timeStr: string): number {
  const parts = timeStr.split(':').map(Number)
  return parts[0] * 3600 + parts[1] * 60 + parts[2]
}

/**
 * Convert total seconds to HH:MM:SS format
 */
export function secondsToTimeFormat(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

/**
 * Parse HH:MM:SS string to total seconds
 * Same as timeToSeconds but with explicit parsing (for clarity in inline usage)
 */
export function parseTimeToSeconds(timeStr: string): number {
  const [h, m, s] = timeStr.split(':').map(Number)
  return h * 3600 + m * 60 + s
}
