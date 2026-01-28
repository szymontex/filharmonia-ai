import { useEffect } from 'react'

interface KeyboardHelpProps {
  show: boolean
  onClose: () => void
}

export default function KeyboardHelp({ show, onClose }: KeyboardHelpProps) {
  // Close on Escape key
  useEffect(() => {
    if (!show) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [show, onClose])

  if (!show) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="space-y-3">
          <ShortcutRow keys="Space" description="Play / Pause" />
          <ShortcutRow keys="Ctrl+S" description="Save" />
          <ShortcutRow keys="Ctrl+Z" description="Undo" />
          <ShortcutRow keys="Ctrl+Shift+Z" description="Redo" />
          <ShortcutRow keys="Ctrl+Y" description="Redo (alt)" />
          <ShortcutRow keys="1" description="Set class: MUSIC" />
          <ShortcutRow keys="2" description="Set class: APPLAUSE" />
          <ShortcutRow keys="3" description="Set class: SPEECH" />
          <ShortcutRow keys="4" description="Set class: PUBLIC" />
          <ShortcutRow keys="5" description="Set class: TUNING" />
          <ShortcutRow keys="?" description="Toggle this help" />
          <ShortcutRow keys="Esc" description="Close this help" />
        </div>

        <div className="mt-6 text-sm text-gray-500 text-center">
          Press <kbd className="bg-gray-100 px-2 py-1 rounded font-mono text-xs">?</kbd> anytime to show this help
        </div>
      </div>
    </div>
  )
}

function ShortcutRow({ keys, description }: { keys: string; description: string }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
      <kbd className="bg-gray-100 px-3 py-1 rounded font-mono text-sm text-gray-800 font-semibold">
        {keys}
      </kbd>
      <span className="text-gray-700 text-sm">{description}</span>
    </div>
  )
}
