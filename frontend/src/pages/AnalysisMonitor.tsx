import { useState, useEffect } from 'react'
import axios from 'axios'
import Toast from '../components/Toast'

interface AnalysisMonitorProps {
  onBack: () => void
}

interface JobStatus {
  status: 'running' | 'completed' | 'failed'
  total: number
  completed: number
  failed: number
  current_file: string | null
  current_file_progress: number
  progress: number
  results: Array<{ mp3: string; csv: string; segments: number }>
  errors: Array<{ mp3: string; error: string }>
}

export default function AnalysisMonitor({ onBack }: AnalysisMonitorProps) {
  const [jobId, setJobId] = useState<string>('')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [availableJobs, setAvailableJobs] = useState<any[]>([])
  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })

  // Load available jobs on mount and refresh
  useEffect(() => {
    loadAvailableJobs()
    const interval = setInterval(loadAvailableJobs, 2000) // Refresh every 2s
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!autoRefresh || !jobId) return

    const interval = setInterval(async () => {
      await fetchJobStatus()
    }, 1000)  // Update every second

    return () => clearInterval(interval)
  }, [autoRefresh, jobId])

  const loadAvailableJobs = async () => {
    try {
      const response = await axios.get('/api/v1/analyze/batch')
      setAvailableJobs(response.data)
    } catch (error) {
      console.error('Error loading jobs:', error)
    }
  }

  const fetchJobStatus = async () => {
    if (!jobId) return

    try {
      setLoading(true)
      const response = await axios.get(`/api/v1/analyze/batch/${jobId}`)
      setJobStatus(response.data)

      // Stop auto-refresh when completed
      if (response.data.status === 'completed') {
        setAutoRefresh(false)
      }
    } catch (error) {
      console.error('Error fetching job status:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleStartMonitoring = () => {
    if (!jobId.trim()) {
      setErrorToast({ show: true, message: 'Please enter a Job ID' })
      return
    }
    setAutoRefresh(true)
    fetchJobStatus()
  }

  const handleCancelJob = async () => {
    if (!jobId) return

    try {
      await axios.post(`/api/v1/analyze/batch/${jobId}/cancel`)
      setSuccessToast({ show: true, message: 'Job cancellation requested' })
      fetchJobStatus()
    } catch (error: any) {
      console.error('Error cancelling job:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Failed to cancel job' })
    }
  }

  const getStatusBadge = (status: string) => {
    const badges: Record<string, { color: string; text: string }> = {
      running: { color: 'bg-blue-100 text-blue-800', text: 'Running' },
      completed: { color: 'bg-green-100 text-green-800', text: 'Completed' },
      failed: { color: 'bg-red-100 text-red-800', text: 'Failed' },
      cancelled: { color: 'bg-orange-100 text-orange-800', text: 'Cancelled' },
      interrupted: { color: 'bg-purple-100 text-purple-800', text: 'Interrupted (Restart)' }
    }
    const badge = badges[status] || { color: 'bg-gray-100 text-gray-800', text: status }
    return (
      <span className={`px-3 py-1 rounded text-sm font-medium ${badge.color}`}>
        {badge.text}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <button onClick={onBack} className="text-blue-600 hover:text-blue-800 mb-2">
            ← Back to Home
          </button>
          <h1 className="text-3xl font-bold">📊 Analysis Monitor</h1>
          <p className="text-gray-600 text-sm mt-1">
            Track batch analysis progress in real-time
          </p>
        </div>

        {/* Available Jobs */}
        {availableJobs.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Recent Jobs</h2>
            <div className="space-y-2">
              {availableJobs.slice(0, 5).map((job) => (
                <div
                  key={job.job_id}
                  onClick={async () => {
                    setJobId(job.job_id)
                    setAutoRefresh(true)
                    // Fetch status immediately
                    try {
                      setLoading(true)
                      const response = await axios.get(`/api/v1/analyze/batch/${job.job_id}`)
                      setJobStatus(response.data)
                    } catch (error) {
                      console.error('Error fetching job status:', error)
                    } finally {
                      setLoading(false)
                    }
                  }}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-blue-50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div>
                      {getStatusBadge(job.status)}
                    </div>
                    <div className="text-sm">
                      <div className="font-medium text-gray-900">
                        {job.type === 'single' ? (
                          <>
                            {job.file ? job.file.split('\\').pop() : 'Single file'}
                            {job.status === 'running' && ` (${job.completed}/${job.total} segments)`}
                          </>
                        ) : (
                          `${job.completed + job.failed} / ${job.total} files`
                        )}
                      </div>
                      <div className="text-xs text-gray-500 font-mono">
                        {job.type === 'single' ? 'Single File' : 'Batch'} • ID: {job.job_id.substring(0, 8)}...
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end">
                    <div className="text-sm font-medium text-gray-600">
                      {job.progress?.toFixed(1) || 0}%
                    </div>
                    {job.status === 'running' && (
                      <div className="w-24 bg-gray-200 rounded-full h-1.5 mt-1">
                        <div
                          className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                          style={{ width: `${job.progress || 0}%` }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Job ID Input (Manual) */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Job ID (Manual Entry)
              </label>
              <input
                type="text"
                value={jobId}
                onChange={(e) => setJobId(e.target.value)}
                placeholder="Or enter Job ID manually"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handleStartMonitoring}
                disabled={loading || autoRefresh}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
              >
                {autoRefresh ? '🔄 Monitoring...' : '▶ Start Monitoring'}
              </button>
            </div>
          </div>
        </div>

        {/* Job Status */}
        {jobStatus && (
          <>
            {/* Overview */}
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Job Status</h2>
                <div className="flex items-center gap-3">
                  {getStatusBadge(jobStatus.status)}
                  {jobStatus.status === 'running' && (
                    <button
                      onClick={handleCancelJob}
                      className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm font-medium"
                    >
                      ✕ Cancel Job
                    </button>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <div className="text-sm text-gray-600">Total Files</div>
                  <div className="text-2xl font-bold">{jobStatus.total}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Completed</div>
                  <div className="text-2xl font-bold text-green-600">{jobStatus.completed}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Failed</div>
                  <div className="text-2xl font-bold text-red-600">{jobStatus.failed}</div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mb-2">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Overall Progress</span>
                  <span>{jobStatus.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                    style={{ width: `${jobStatus.progress}%` }}
                  />
                </div>
              </div>

              {/* Current File */}
              {jobStatus.current_file && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Processing:</div>
                  <div className="font-medium text-gray-900 mb-2">{jobStatus.current_file}</div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all duration-150"
                      style={{ width: `${jobStatus.current_file_progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Results */}
            {jobStatus.results.length > 0 && (
              <div className="bg-white rounded-lg shadow p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4">✅ Completed ({jobStatus.results.length})</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {jobStatus.results.map((result, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-green-50 rounded">
                      <div className="flex-1">
                        <div className="font-medium text-sm">{result.mp3.split('\\').pop()}</div>
                        <div className="text-xs text-gray-600">{result.segments} segments</div>
                      </div>
                      <div className="text-xs text-gray-500">
                        CSV: {result.csv.split('\\').pop()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Errors */}
            {jobStatus.errors.length > 0 && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">❌ Errors ({jobStatus.errors.length})</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {jobStatus.errors.map((error, idx) => (
                    <div key={idx} className="p-3 bg-red-50 rounded">
                      <div className="font-medium text-sm text-red-900">{error.mp3.split('\\').pop()}</div>
                      <div className="text-xs text-red-600">{error.error}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {!jobStatus && !loading && (
          <div className="text-center py-12 text-gray-500">
            Enter a Job ID to start monitoring
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
        autoClose={3000}
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
    </div>
  )
}
