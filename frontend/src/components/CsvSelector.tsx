interface CsvFile {
  path: string
  name: string
  date: string
}

interface CsvSelectorProps {
  files: CsvFile[]
  selectedCsv: string | null
  onSelect: (path: string) => void
  onDelete: (path: string, event: React.MouseEvent) => void
  analyzingFiles: Map<string, number>
  editedCsvs: Set<string>
  csvsWithExports: Set<string>
}

/**
 * CSV file selector component
 * Displays list of analysis result files with status badges
 */
export function CsvSelector({
  files,
  selectedCsv,
  onSelect,
  onDelete,
  analyzingFiles,
  editedCsvs,
  csvsWithExports
}: CsvSelectorProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-semibold mb-4">Select Analysis Result</h2>
      <div className="grid grid-cols-1 gap-3 max-h-60 overflow-y-auto">
        {files.map(file => {
          // Extract song name from predictions_SONG042_2025-09-27.csv
          const songMatch = file.name.match(/predictions_(.+?)_\d{4}-\d{2}-\d{2}/)
          const songName = songMatch ? songMatch[1] : file.name

          // Extract time from filename (if present)
          const timeMatch = file.name.match(/_(\d{2})-(\d{2})\.csv$/)
          const timeStr = timeMatch ? `${timeMatch[1]}:${timeMatch[2]}` : null

          // Format date as DD.MM.YYYY
          const dateParts = file.date.split('-')
          const formattedDate = dateParts.length === 3
            ? `${dateParts[2]}.${dateParts[1]}.${dateParts[0]}`
            : file.date

          // Check if this file is currently being analyzed
          const isAnalyzing = analyzingFiles.has(songName + '.MP3') || analyzingFiles.has(songName + '.mp3')
          const analyzingProgress = (analyzingFiles.get(songName + '.MP3') || analyzingFiles.get(songName + '.mp3')) ?? 0

          return (
            <div
              key={file.path}
              className={`relative p-4 rounded border hover:bg-blue-50 cursor-pointer ${
                selectedCsv === file.path ? 'bg-blue-100 border-blue-500' : 'border-gray-200'
              }`}
              onClick={() => onSelect(file.path)}
            >
              <div className="flex items-center gap-2 mb-1">
                <div className="text-xl font-bold text-blue-600">{formattedDate}</div>
                {timeStr && (
                  <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-sm font-semibold rounded">
                    {timeStr}
                  </span>
                )}
                {editedCsvs.has(file.path) && (
                  <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs font-semibold rounded">
                    EDITED
                  </span>
                )}
                {csvsWithExports.has(file.path) && (
                  <span className="px-2 py-0.5 bg-purple-100 text-purple-800 text-xs font-semibold rounded">
                    EXPORTED
                  </span>
                )}
                {isAnalyzing && (
                  <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded animate-pulse">
                    Analyzing {analyzingProgress.toFixed(0)}%
                  </span>
                )}
              </div>
              <div className="text-sm font-medium text-gray-800">{songName}</div>
              <div className="text-xs text-gray-500 mt-1">{file.name}</div>
              <button
                onClick={(e) => onDelete(file.path, e)}
                className="absolute top-3 right-3 text-red-500 hover:text-red-700 hover:bg-red-100 rounded px-2 py-1 text-lg"
                title="Delete CSV"
              >
                ×
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
