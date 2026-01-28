import { useEffect, useRef } from 'react'

/**
 * Global keyboard shortcut handler hook
 *
 * Registers a single keydown listener on document that calls handlers based on key combinations.
 * Skips when user is typing in input fields.
 *
 * @param handlers - Map of key combinations to handler functions
 *                   Format: "ctrl+z", "ctrl+shift+z", "space", etc.
 *
 * @example
 * useKeyboardShortcuts({
 *   'ctrl+z': handleUndo,
 *   'ctrl+shift+z': handleRedo,
 *   'space': togglePlayPause
 * })
 */
export function useKeyboardShortcuts(handlers: Record<string, () => void>): void {
  // Use ref to avoid re-registering listener on every render
  const handlersRef = useRef(handlers)

  // Keep ref current
  useEffect(() => {
    handlersRef.current = handlers
  })

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement

      // Skip if user is typing in text input
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.contentEditable === 'true'
      ) {
        return
      }

      // Build key combination string
      let combo = ''

      if (e.ctrlKey || e.metaKey) {
        combo += 'ctrl+'
      }

      if (e.shiftKey) {
        combo += 'shift+'
      }

      // Normalize space key
      if (e.key === ' ') {
        combo += 'space'
      } else {
        combo += e.key.toLowerCase()
      }

      // Check if we have a handler for this combo
      const handler = handlersRef.current[combo]

      if (handler) {
        e.preventDefault()
        handler()
      }
    }

    // Register listener once
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, []) // Empty deps - listener registered once
}
