import { CLASS_COLORS } from '../constants/colors'
import { Track } from '../hooks/useTrackEditor'

interface TrackTableProps {
  tracks: Track[]
  exportedSegments: Set<number>
  playingTrackId: string | null
  selectedTrackId: string | null

  // Track operations
  onToggleSelect: (id: string) => void
  onUpdateName: (id: string, name: string) => void
  onNameFocus?: () => void
  onUpdateClass: (id: string, predicted_class: string) => void
  onUpdateStart: (id: string, start: string) => void
  onUpdateStop: (id: string, stop: string) => void
  onDelete: (id: string) => void
  onMergeWithNext: (id: string) => void
  onAddSegmentBelow: (id: string) => void

  // Selection and playback
  onSelectTrack: (id: string | null) => void
  onPlayFrom: (startTime: string, trackId: string) => void
  onUndoExport: (segmentIndex: number) => void

  // Header checkbox
  onSelectAll: (selected: boolean) => void
}

function getClassColor(cls: string): string {
  const config = CLASS_COLORS[cls as keyof typeof CLASS_COLORS]
  if (!config) return 'bg-gray-100 text-gray-800'
  return `${config.bg} ${config.text}`
}

/**
 * Track table component
 * Renders editable track rows with actions
 */
export function TrackTable({
  tracks,
  exportedSegments,
  playingTrackId,
  selectedTrackId,
  onToggleSelect,
  onUpdateName,
  onNameFocus,
  onUpdateClass,
  onUpdateStart,
  onUpdateStop,
  onDelete,
  onMergeWithNext,
  onAddSegmentBelow,
  onSelectTrack,
  onPlayFrom,
  onUndoExport,
  onSelectAll
}: TrackTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-gray-50 border-b">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={tracks.length > 0 && tracks.every(t => t.selected)}
                  onChange={(e) => onSelectAll(e.target.checked)}
                  className="w-4 h-4 cursor-pointer"
                />
                <span>Select</span>
              </div>
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Start</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Stop</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Play</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {tracks.map((track, idx) => {
            const isExported = exportedSegments.has(idx)
            const isPlaying = playingTrackId === track.id
            const isHovered = selectedTrackId === track.id

            let bgColor = ''
            let hoverClass = 'hover:bg-blue-100'

            if (isPlaying) {
              bgColor = 'bg-green-100 border-l-4 border-green-500'
              hoverClass = 'hover:bg-green-200'
            } else if (isHovered) {
              bgColor = 'bg-blue-100'
            } else if (isExported) {
              bgColor = 'bg-purple-50'
            }

            return (
              <tr
                key={track.id}
                onMouseEnter={() => onSelectTrack(track.id)}
                onMouseLeave={() => onSelectTrack(null)}
                className={`cursor-pointer transition-colors ${bgColor} ${hoverClass}`}
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onToggleSelect(track.id)}
                      className="text-xl"
                    >
                      {track.selected ? '\u2713' : '\u2717'}
                    </button>
                    {isExported && (
                      <div className="flex items-center gap-1">
                        <span
                          className="px-1.5 py-0.5 bg-purple-100 text-purple-800 text-xs font-semibold rounded"
                          title="Already exported to training data"
                        >
                          EXP
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onUndoExport(idx)
                          }}
                          className="px-1.5 py-0.5 bg-red-100 text-red-700 text-xs font-semibold rounded hover:bg-red-200"
                          title="Undo export - delete from training data"
                        >
                          Undo
                        </button>
                      </div>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <input
                    type="text"
                    value={track.name}
                    onChange={(e) => onUpdateName(track.id, e.target.value)}
                    onFocus={onNameFocus}
                    className="w-full px-2 py-1 border rounded"
                    placeholder="Track name..."
                  />
                </td>
                <td className="px-4 py-3">
                  <select
                    value={track.predicted_class}
                    onChange={(e) => onUpdateClass(track.id, e.target.value)}
                    className={`px-2 py-1 rounded text-sm font-medium border ${getClassColor(track.predicted_class)}`}
                  >
                    {Object.keys(CLASS_COLORS).map(cls => (
                      <option key={cls} value={cls}>{cls}</option>
                    ))}
                  </select>
                </td>
                <td className="px-4 py-3">
                  <input
                    type="text"
                    value={track.start}
                    onChange={(e) => onUpdateStart(track.id, e.target.value)}
                    className="w-20 px-2 py-1 border rounded text-sm"
                    placeholder="HH:MM:SS"
                  />
                </td>
                <td className="px-4 py-3">
                  <input
                    type="text"
                    value={track.stop}
                    onChange={(e) => onUpdateStop(track.id, e.target.value)}
                    className="w-20 px-2 py-1 border rounded text-sm"
                    placeholder="HH:MM:SS"
                  />
                </td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => onPlayFrom(track.start, track.id)}
                    className="px-2 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                  >
                    Play
                  </button>
                </td>
                <td className="px-4 py-3 text-sm font-medium">{track.duration}</td>
                <td className="px-4 py-3">
                  <div className="flex gap-2">
                    <button
                      onClick={() => onAddSegmentBelow(track.id)}
                      className="px-2 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                      title="Add new segment below this one"
                    >
                      + Below
                    </button>
                    <button
                      onClick={() => onMergeWithNext(track.id)}
                      className="px-2 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                    >
                      Merge
                    </button>
                    <button
                      onClick={() => onDelete(track.id)}
                      className="px-2 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
