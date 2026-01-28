interface PlayerControlsProps {
  // Player state
  showPlayer: boolean
  mp3Path: string
  onTogglePlayer: () => void

  // Save/discard
  hasUnsavedChanges: boolean
  onSave: () => void
  onDiscard: () => void

  // Threshold
  threshold: number
  onThresholdChange: (value: number) => void
  thresholdDisabled: boolean

  // Export
  selectedCount: number
  onExportToTraining: () => void
  onCopyTracklist: () => void

  // Progress indicator
  progressStage?: string | null
}

/**
 * Player and action controls bar
 * Includes threshold slider, player toggle, save/discard, and export buttons
 */
export function PlayerControls({
  showPlayer,
  mp3Path,
  onTogglePlayer,
  hasUnsavedChanges,
  onSave,
  onDiscard,
  threshold,
  onThresholdChange,
  thresholdDisabled,
  selectedCount,
  onExportToTraining,
  onCopyTracklist,
  progressStage
}: PlayerControlsProps) {
  return (
    <div className="flex items-center gap-4">
      {/* Progress Indicator */}
      {progressStage && (
        <div className="flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-800 rounded-md animate-pulse">
          <span className="text-sm font-medium">{progressStage}</span>
        </div>
      )}
      {/* Threshold Slider */}
      <div className="flex items-center gap-3 mr-4 bg-gray-50 px-4 py-2 rounded-lg border border-gray-200">
        <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
          Noise Filter:
        </label>
        <input
          type="range"
          min="1"
          max="15"
          value={threshold}
          onChange={(e) => onThresholdChange(parseInt(e.target.value))}
          className="w-32"
          disabled={thresholdDisabled}
        />
        <span className="text-sm font-semibold text-blue-600 min-w-[4rem]">
          {threshold} segs
        </span>
      </div>

      <button
        onClick={onTogglePlayer}
        disabled={!mp3Path}
        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-300"
      >
        {showPlayer ? 'Hide Player' : 'Show Player'}
      </button>

      <button
        onClick={onSave}
        disabled={!hasUnsavedChanges}
        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-300"
      >
        Save
      </button>

      <button
        onClick={onDiscard}
        disabled={!hasUnsavedChanges}
        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-300"
      >
        Discard
      </button>

      <button
        onClick={onExportToTraining}
        disabled={selectedCount === 0}
        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-300"
        title="Export selected segments to TRAINING DATA folder"
      >
        Export to Training
      </button>

      <button
        onClick={onCopyTracklist}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        title="Copy tracklist to clipboard (MUSIC segments only)"
      >
        Copy Tracklist
      </button>
    </div>
  )
}
