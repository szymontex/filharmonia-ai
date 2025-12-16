import { useState, useEffect } from 'react'
import axios from 'axios'
import Toast from '../components/Toast'

interface Recording {
  path: string
  name: string
  size: number
  date: string  // YYYY-MM-DD format
  time?: string  // HH:MM from ID3 tag
}

interface CalendarBrowserProps {
  onBack: () => void
  onOpenCsv?: (csvPath: string) => void
}

export default function CalendarBrowser({ onBack, onOpenCsv }: CalendarBrowserProps) {
  const [recordings, setRecordings] = useState<Recording[]>([])
  const [selectedYear, setSelectedYear] = useState<string>('')
  const [selectedMonth, setSelectedMonth] = useState<string>('')
  const [loading, setLoading] = useState(true)

  // Load recordings on mount and when filters change
  useEffect(() => {
    loadRecordings()
  }, [])

  const loadRecordings = async () => {
    try {
      setLoading(true)
      const [recordingsRes, csvRes] = await Promise.all([
        axios.get('/api/v1/files/sorted'),
        axios.get('/api/v1/files/analysis-results')
      ])
      setRecordings(recordingsRes.data)

      // Extract analyzed files as "stem_date" to avoid false positives (e.g., SONG005 from 2023 vs 2025)
      const analyzed = new Set<string>()
      csvRes.data.forEach((csv: any) => {
        // Extract SONG name and date from predictions_SONG042_2025-09-27.csv
        const match = csv.name.match(/predictions_(.+?)_(\d{4}-\d{2}-\d{2})/)
        if (match) {
          analyzed.add(`${match[1]}_${match[2]}`)  // SONG042_2025-09-27
        }
      })
      setAnalyzedFiles(analyzed)

      // Set default year/month to most recent recording if not set
      if (!selectedYear && recordingsRes.data.length > 0) {
        const latestDate = recordingsRes.data[0].date  // Already sorted newest first
        const [year, month] = latestDate.split('-')
        setSelectedYear(year)
        setSelectedMonth(month.replace(/^0/, ''))  // Remove leading zero: '09' -> '9'
      }
    } catch (error) {
      console.error('Error loading recordings:', error)
    } finally {
      setLoading(false)
    }
  }

  // Get unique years from recordings
  const availableYears = Array.from(
    new Set(recordings.map(r => r.date.split('-')[0]))
  ).sort().reverse()

  // Filter recordings by selected year/month
  const filteredRecordings = recordings.filter(r => {
    const [year, month] = r.date.split('-')
    const yearMatches = selectedYear === 'all' || year === selectedYear
    const monthMatches = selectedMonth === 'all' || month === selectedMonth.padStart(2, '0')
    return yearMatches && monthMatches
  })

  // Group recordings by date
  const groupedByDate = filteredRecordings.reduce((acc, rec) => {
    const date = rec.date
    if (!acc[date]) acc[date] = []
    acc[date].push(rec)
    return acc
  }, {} as Record<string, Recording[]>)

  const sortedDates = Object.keys(groupedByDate).sort().reverse()

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024)
    return mb.toFixed(1) + ' MB'
  }

  const formatDate = (dateStr: string) => {
    const [year, month, day] = dateStr.split('-')
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return `${day} ${monthNames[parseInt(month) - 1]} ${year}`
  }

  const [analyzing, setAnalyzing] = useState<Set<string>>(new Set())  // paths of files being analyzed
  const [analyzingProgress, setAnalyzingProgress] = useState<Map<string, number>>(new Map())  // path -> progress %
  const [analyzedFiles, setAnalyzedFiles] = useState<Set<string>>(new Set())  // paths of analyzed files
  const [batchJobId, setBatchJobId] = useState<string | null>(null)
  const [batchProgress, setBatchProgress] = useState<any>(null)
  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [batchConfirm, setBatchConfirm] = useState<{ show: boolean; count: number }>({ show: false, count: 0 })
  const [reanalyzeExisting, setReanalyzeExisting] = useState<boolean>(false)

  const handlePlayRecording = (recording: Recording) => {
    // TODO: Open player/editor
    console.log('Play recording:', recording)
  }

  const handleAnalyze = async (recording: Recording) => {
    try {
      // Add to analyzing set
      setAnalyzing(prev => new Set(prev).add(recording.path))

      const response = await axios.post('/api/v1/analyze/', {
        mp3_path: recording.path
      })

      const jobId = response.data.job_id

      // Monitor job status
      const checkStatus = async () => {
        try {
          const statusRes = await axios.get(`/api/v1/analyze/status/${jobId}`)

          if (statusRes.data.status === 'completed') {
            // Remove from analyzing set and progress
            setAnalyzing(prev => {
              const next = new Set(prev)
              next.delete(recording.path)
              return next
            })
            setAnalyzingProgress(prev => {
              const next = new Map(prev)
              next.delete(recording.path)
              return next
            })

            // Add to analyzed files set (avoid full reload)
            const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
            const key = `${stem}_${recording.date}`
            setAnalyzedFiles(prev => new Set(prev).add(key))

            setSuccessToast({
              show: true,
              message: `Analysis complete! ${statusRes.data.segments_analyzed} segments analyzed in ${Math.round(statusRes.data.duration_seconds / 60)} min`
            })
          } else if (statusRes.data.status === 'failed') {
            // Remove from analyzing set and progress
            setAnalyzing(prev => {
              const next = new Set(prev)
              next.delete(recording.path)
              return next
            })
            setAnalyzingProgress(prev => {
              const next = new Map(prev)
              next.delete(recording.path)
              return next
            })
            setErrorToast({ show: true, message: statusRes.data.error || 'Analysis failed' })
          } else {
            // Still running - update progress
            const progress = statusRes.data.progress || 0
            setAnalyzingProgress(prev => {
              const next = new Map(prev)
              next.set(recording.path, progress)
              return next
            })
            // Check again
            setTimeout(checkStatus, 1000)
          }
        } catch (error) {
          console.error('Error checking status:', error)
          // Remove from analyzing set and progress
          setAnalyzing(prev => {
            const next = new Set(prev)
            next.delete(recording.path)
            return next
          })
          setAnalyzingProgress(prev => {
            const next = new Map(prev)
            next.delete(recording.path)
            return next
          })
        }
      }

      setTimeout(checkStatus, 1000)
    } catch (error: any) {
      console.error('Analysis error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Analysis failed' })
      // Remove from analyzing set
      setAnalyzing(prev => {
        const next = new Set(prev)
        next.delete(recording.path)
        return next
      })
    }
  }

  const handleBatchAnalyze = () => {
    const recordingsToAnalyze = reanalyzeExisting
      ? filteredRecordings
      : filteredRecordings.filter(r => !isAnalyzed(r))

    if (recordingsToAnalyze.length === 0) {
      setErrorToast({ show: true, message: 'All recordings already analyzed! Check "Re-analyze existing" to re-analyze.' })
      return
    }

    setBatchConfirm({ show: true, count: recordingsToAnalyze.length })
  }

  const performBatchAnalyze = async () => {
    setBatchConfirm({ show: false, count: 0 })

    try {
      const recordingsToAnalyze = reanalyzeExisting
        ? filteredRecordings
        : filteredRecordings.filter(r => !isAnalyzed(r))

      const paths = recordingsToAnalyze.map(r => r.path)
      const response = await axios.post('/api/v1/analyze/batch', {
        mp3_paths: paths
      })
      setBatchJobId(response.data.job_id)
      startBatchMonitoring(response.data.job_id)
    } catch (error: any) {
      console.error('Batch analyze error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Batch analyze failed' })
    }
  }

  const startBatchMonitoring = (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`/api/v1/analyze/batch/${jobId}`)
        setBatchProgress(response.data)

        if (response.data.status === 'completed') {
          clearInterval(interval)
          setBatchJobId(null)
          setBatchProgress(null)
          loadRecordings()  // Refresh
          setSuccessToast({
            show: true,
            message: `Batch analysis complete! Completed: ${response.data.completed} • Failed: ${response.data.failed}`
          })
        }
      } catch (error) {
        console.error('Error fetching batch status:', error)
        clearInterval(interval)
      }
    }, 1000)
  }

  const isAnalyzed = (recording: Recording) => {
    const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
    const key = `${stem}_${recording.date}`  // e.g., SONG005_2023-09-24
    return analyzedFiles.has(key)
  }

  const getCsvPath = (recording: Recording) => {
    const stem = recording.name.replace('.MP3', '').replace('.mp3', '')
    const date = recording.date
    return `Y:\\!_FILHARMONIA\\SORTED\\ANALYSIS_RESULTS\\predictions_${stem}_${date}.csv`
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <button
              onClick={onBack}
              className="text-blue-600 hover:text-blue-800 mb-2"
            >
              ← Back to Home
            </button>
            <h1 className="text-3xl font-bold">📅 Browse Recordings</h1>
            <p className="text-gray-600 text-sm mt-1">
              {recordings.length} recordings in SORTED folder
            </p>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={reanalyzeExisting}
                onChange={(e) => setReanalyzeExisting(e.target.checked)}
                className="w-4 h-4"
              />
              Re-analyze existing
            </label>
            <button
              onClick={handleBatchAnalyze}
              disabled={!!batchJobId || filteredRecordings.length === 0}
              className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400"
            >
              ⚡ Batch Analyze All ({reanalyzeExisting ? filteredRecordings.length : filteredRecordings.filter(r => !isAnalyzed(r)).length})
            </button>
            <button
              onClick={loadRecordings}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              🔄 Refresh
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow p-4 mb-6">
          <div className="flex gap-4 items-center">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Year
              </label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                {availableYears.map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
                <option value="all">All Years</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Month
              </label>
              <select
                value={selectedMonth}
                onChange={(e) => setSelectedMonth(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Months</option>
                <option value="1">January</option>
                <option value="2">February</option>
                <option value="3">March</option>
                <option value="4">April</option>
                <option value="5">May</option>
                <option value="6">June</option>
                <option value="7">July</option>
                <option value="8">August</option>
                <option value="9">September</option>
                <option value="10">October</option>
                <option value="11">November</option>
                <option value="12">December</option>
              </select>
            </div>

            <div className="ml-auto text-sm text-gray-600">
              Showing {filteredRecordings.length} recording(s)
            </div>
          </div>
        </div>

        {/* Recordings List */}
        {loading ? (
          <div className="text-center py-12 text-gray-500">
            Loading recordings...
          </div>
        ) : sortedDates.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            No recordings found for {selectedYear} {selectedMonth !== 'all' ? `/ ${selectedMonth}` : ''}
          </div>
        ) : (
          <div className="space-y-6">
            {sortedDates.map(date => (
              <div key={date} className="bg-white rounded-lg shadow overflow-hidden">
                <div className="bg-gray-100 px-6 py-3 border-b">
                  <h2 className="text-lg font-semibold text-gray-800">
                    {formatDate(date)}
                  </h2>
                </div>
                <div className="divide-y">
                  {groupedByDate[date].map((recording, idx) => (
                    <div
                      key={idx}
                      className="px-6 py-4 hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">🎵</span>
                            <div>
                              <div className="flex items-center gap-2">
                                <h3 className="font-medium text-gray-900">
                                  {recording.name}
                                </h3>
                                {recording.time && (
                                  <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs font-semibold rounded">
                                    {recording.time}
                                  </span>
                                )}
                                {analyzing.has(recording.path) && (
                                  <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded animate-pulse">
                                    ⏳ Analyzing {analyzingProgress.get(recording.path)?.toFixed(0) || 0}%
                                  </span>
                                )}
                                {batchProgress?.current_file === recording.name && (
                                  <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded animate-pulse">
                                    ⏳ Analyzing {batchProgress.current_file_progress.toFixed(0)}%
                                  </span>
                                )}
                              </div>
                              <p className="text-sm text-gray-500">
                                {formatFileSize(recording.size)}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-2">
                          <button
                            onClick={() => handlePlayRecording(recording)}
                            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
                          >
                            ▶ Play
                          </button>
                          {isAnalyzed(recording) ? (
                            <>
                              <button
                                onClick={() => handleAnalyze(recording)}
                                disabled={analyzing.has(recording.path)}
                                className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm disabled:bg-gray-400"
                              >
                                {analyzing.has(recording.path) ? '⏳ Re-analyzing...' : '🔄 Re-analyze'}
                              </button>
                              {onOpenCsv && (
                                <button
                                  onClick={() => onOpenCsv(getCsvPath(recording))}
                                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
                                >
                                  📝 Edit CSV
                                </button>
                              )}
                            </>
                          ) : (
                            <button
                              onClick={() => handleAnalyze(recording)}
                              disabled={analyzing.has(recording.path)}
                              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:bg-gray-400"
                            >
                              {analyzing.has(recording.path) ? '⏳ Analyzing...' : '📊 Analyze'}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Floating Progress Bar */}
        {batchProgress && (
          <div className="fixed bottom-0 left-0 right-0 bg-white border-t-2 border-blue-500 shadow-lg p-4">
            <div className="max-w-7xl mx-auto">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div className="text-lg font-semibold">
                    Batch Analysis in Progress
                  </div>
                  <div className="text-sm text-gray-600">
                    {batchProgress.completed + batchProgress.failed} / {batchProgress.total} files
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  {batchProgress.progress}%
                </div>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${batchProgress.progress}%` }}
                />
              </div>

              {batchProgress.current_file && (
                <div className="text-sm text-gray-700">
                  Processing: <span className="font-medium">{batchProgress.current_file}</span>
                  <span className="ml-2 text-gray-500">({batchProgress.current_file_progress.toFixed(0)}%)</span>
                </div>
              )}

              <div className="mt-2 flex gap-4 text-sm">
                <span className="text-green-600">✓ Completed: {batchProgress.completed}</span>
                <span className="text-red-600">✗ Failed: {batchProgress.failed}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Success Toast */}
      <Toast
        show={successToast.show}
        onClose={() => setSuccessToast({ show: false, message: '' })}
        title="Success!"
        message={successToast.message}
        icon="✅"
        color="green"
        index={0}
        autoClose={5000}
      />

      {/* Error Toast */}
      <Toast
        show={errorToast.show}
        onClose={() => setErrorToast({ show: false, message: '' })}
        title="Error"
        message={errorToast.message}
        icon="❌"
        color="red"
        index={successToast.show ? 1 : 0}
        autoClose={5000}
      />

      {/* Batch Analyze Confirmation */}
      <Toast
        show={batchConfirm.show}
        onClose={() => setBatchConfirm({ show: false, count: 0 })}
        title="Batch Analyze All Recordings?"
        message={`Analyze ${batchConfirm.count} recording${batchConfirm.count !== 1 ? 's' : ''} in current view?`}
        icon="⚡"
        color="blue"
        index={(successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0)}
        autoClose={0}
        actions={[
          {
            label: 'Analyze',
            onClick: performBatchAnalyze,
            color: 'primary'
          },
          {
            label: 'Cancel',
            onClick: () => setBatchConfirm({ show: false, count: 0 }),
            color: 'secondary'
          }
        ]}
      />
    </div>
  )
}
