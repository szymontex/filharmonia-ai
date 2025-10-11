import { useState } from 'react'
import axios from 'axios'
import Toast from '../components/Toast'

interface FileToSort {
  path: string
  name: string
  size: number
  date?: string
  target_path?: string
  existing_path?: string
  existing_size?: number
  status: 'ready_to_move' | 'duplicate' | 'exists_different_size' | 'error' | 'invalid_date' | 'no_metadata'
  error?: string
}

interface SortManagerProps {
  onBack: () => void
}

export default function SortManager({ onBack }: SortManagerProps) {
  const [files, setFiles] = useState<FileToSort[]>([])
  const [scanning, setScanning] = useState(false)
  const [sorting, setSorting] = useState(false)
  const [sortProgress, setSortProgress] = useState({ current: 0, total: 0 })
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [stats, setStats] = useState({ total: 0, ready_to_move: 0, duplicates: 0, errors: 0 })
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [sortConfirm, setSortConfirm] = useState<{ show: boolean; count: number }>({ show: false, count: 0 })
  const [sortSuccess, setSortSuccess] = useState<{ show: boolean; moved: number; duplicates: number; renamed: number; errors: number; movedPaths: string[] }>({ show: false, moved: 0, duplicates: 0, renamed: 0, errors: 0, movedPaths: [] })
  const [deleteDuplicatesConfirm, setDeleteDuplicatesConfirm] = useState<{ show: boolean; count: number }>({ show: false, count: 0 })

  const handleScan = async () => {
    try {
      setScanning(true)
      const response = await axios.get('http://localhost:8000/api/v1/sort/scan')
      setFiles(response.data.files)
      setStats({
        total: response.data.total,
        ready_to_move: response.data.ready_to_move,
        duplicates: response.data.duplicates,
        errors: response.data.errors
      })
      // Auto-select ready_to_move files
      const readyFiles = new Set(
        response.data.files
          .filter((f: FileToSort) => f.status === 'ready_to_move')
          .map((f: FileToSort) => f.path)
      )
      setSelectedFiles(readyFiles)
    } catch (error) {
      console.error('Scan error:', error)
      setErrorToast({ show: true, message: 'Error scanning files' })
    } finally {
      setScanning(false)
    }
  }

  const [sortedFiles, setSortedFiles] = useState<string[]>([])  // Paths of sorted files

  const handleSort = () => {
    if (selectedFiles.size === 0) {
      setErrorToast({ show: true, message: 'No files selected' })
      return
    }
    setSortConfirm({ show: true, count: selectedFiles.size })
  }

  const performSort = async () => {
    setSortConfirm({ show: false, count: 0 })

    try {
      setSorting(true)
      const filePaths = Array.from(selectedFiles)
      setSortProgress({ current: 0, total: filePaths.length })

      const response = await axios.post('http://localhost:8000/api/v1/sort/execute', {
        file_paths: filePaths
      })

      const summary = response.data.summary

      // Save moved file paths for batch analyze
      const movedPaths = response.data.moved.map((m: any) => m.target)
      setSortedFiles(movedPaths)
      setSortProgress({ current: filePaths.length, total: filePaths.length })

      setSortSuccess({
        show: true,
        moved: summary.moved,
        duplicates: summary.duplicates_removed,
        renamed: summary.renamed,
        errors: summary.errors,
        movedPaths
      })

      // Refresh list
      handleScan()
    } catch (error: any) {
      console.error('Sort error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Sort failed' })
    } finally {
      setSorting(false)
      setSortProgress({ current: 0, total: 0 })
    }
  }

  const handleBatchAnalyze = async (paths: string[]) => {
    setSortSuccess({ show: false, moved: 0, duplicates: 0, renamed: 0, errors: 0, movedPaths: [] })

    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze/batch', {
        mp3_paths: paths
      })
      setSuccessToast({
        show: true,
        message: `Batch analysis started! Job ID: ${response.data.job_id} • ${response.data.files_queued} files queued`
      })
    } catch (error: any) {
      console.error('Batch analyze error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Batch analyze failed' })
    }
  }

  const toggleFile = (path: string) => {
    const newSelected = new Set(selectedFiles)
    if (newSelected.has(path)) {
      newSelected.delete(path)
    } else {
      newSelected.add(path)
    }
    setSelectedFiles(newSelected)
  }

  const toggleAll = () => {
    if (selectedFiles.size === filteredFiles.length && filteredFiles.length > 0) {
      setSelectedFiles(new Set())
    } else {
      setSelectedFiles(new Set(filteredFiles.map(f => f.path)))
    }
  }

  const handleDeleteDuplicates = () => {
    const duplicates = files.filter(f => f.status === 'duplicate' && selectedFiles.has(f.path))

    if (duplicates.length === 0) {
      setErrorToast({ show: true, message: 'No duplicates selected' })
      return
    }

    setDeleteDuplicatesConfirm({ show: true, count: duplicates.length })
  }

  const performDeleteDuplicates = async () => {
    const duplicates = files.filter(f => f.status === 'duplicate' && selectedFiles.has(f.path))
    setDeleteDuplicatesConfirm({ show: false, count: 0 })

    try {
      const response = await axios.post('http://localhost:8000/api/v1/sort/delete-duplicates', {
        file_paths: duplicates.map(f => f.path)
      })

      setSuccessToast({
        show: true,
        message: `Deleted ${response.data.deleted_count} duplicates${response.data.error_count > 0 ? ` • Errors: ${response.data.error_count}` : ''}`
      })

      // Refresh list
      handleScan()
    } catch (error: any) {
      console.error('Delete error:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || error.message || 'Delete failed' })
    }
  }

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024)
    return mb.toFixed(1) + ' MB'
  }

  const filteredFiles = statusFilter === 'all'
    ? files
    : files.filter(f => f.status === statusFilter)

  const getStatusBadge = (status: string) => {
    const badges: Record<string, { color: string; text: string }> = {
      ready_to_move: { color: 'bg-green-100 text-green-800', text: 'Ready' },
      duplicate: { color: 'bg-yellow-100 text-yellow-800', text: 'Duplicate' },
      exists_different_size: { color: 'bg-orange-100 text-orange-800', text: 'Exists (diff size)' },
      error: { color: 'bg-red-100 text-red-800', text: 'Error' },
      invalid_date: { color: 'bg-red-100 text-red-800', text: 'Invalid date' },
      no_metadata: { color: 'bg-gray-100 text-gray-800', text: 'No metadata' }
    }
    const badge = badges[status] || { color: 'bg-gray-100 text-gray-800', text: status }
    return (
      <span className={`px-2 py-1 rounded text-xs font-medium ${badge.color}`}>
        {badge.text}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <button onClick={onBack} className="text-blue-600 hover:text-blue-800 mb-2">
              ← Back to Home
            </button>
            <h1 className="text-3xl font-bold">🗂️ Sort Manager</h1>
            <p className="text-gray-600 text-sm mt-1">
              Scan and organize files from !NAGRANIA KONCERTÓW
            </p>
          </div>

          <button
            onClick={handleScan}
            disabled={scanning}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium disabled:bg-gray-400"
          >
            {scanning ? '🔄 Scanning...' : '🔍 Scan for New Files'}
          </button>
        </div>

        {/* Stats */}
        {stats.total > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <div className="grid grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-gray-600">Total Files</div>
                <div className="text-2xl font-bold">{stats.total}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Ready to Move</div>
                <div className="text-2xl font-bold text-green-600">{stats.ready_to_move}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Duplicates</div>
                <div className="text-2xl font-bold text-yellow-600">{stats.duplicates}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">Errors</div>
                <div className="text-2xl font-bold text-red-600">{stats.errors}</div>
              </div>
            </div>
          </div>
        )}

        {/* File List */}
        {files.length > 0 && (
          <>
            <div className="bg-white rounded-lg shadow mb-4">
              <div className="px-6 py-4 border-b flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <input
                    type="checkbox"
                    checked={selectedFiles.size === filteredFiles.length && filteredFiles.length > 0}
                    onChange={toggleAll}
                    className="w-4 h-4"
                  />
                  <span className="font-medium">
                    {selectedFiles.size} / {filteredFiles.length} selected
                  </span>
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="px-3 py-1 border border-gray-300 rounded text-sm"
                  >
                    <option value="all">All ({files.length})</option>
                    <option value="ready_to_move">Ready to Move ({stats.ready_to_move})</option>
                    <option value="duplicate">Duplicates ({stats.duplicates})</option>
                    <option value="exists_different_size">Exists (diff size)</option>
                    <option value="error">Errors ({stats.errors})</option>
                    <option value="invalid_date">Invalid Date</option>
                    <option value="no_metadata">No Metadata</option>
                  </select>
                </div>
                <div className="flex items-center gap-3">
                  {sorting && sortProgress.total > 0 && (
                    <div className="text-sm text-gray-600">
                      {sortProgress.current}/{sortProgress.total} files
                    </div>
                  )}
                  <button
                    onClick={handleDeleteDuplicates}
                    disabled={files.filter(f => f.status === 'duplicate' && selectedFiles.has(f.path)).length === 0}
                    className="px-6 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:bg-gray-400"
                  >
                    🗑️ Delete Duplicates
                  </button>
                  <button
                    onClick={handleSort}
                    disabled={sorting || selectedFiles.size === 0}
                    className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400"
                  >
                    {sorting ? '⏳ Sorting...' : `✓ Sort ${selectedFiles.size} File(s)`}
                  </button>
                </div>
              </div>

              <div className="divide-y max-h-[600px] overflow-y-auto">
                {filteredFiles.map((file, idx) => (
                  <div
                    key={idx}
                    className={`px-6 py-4 hover:bg-gray-50 ${selectedFiles.has(file.path) ? 'bg-blue-50' : ''}`}
                  >
                    <div className="flex items-start gap-4">
                      <input
                        type="checkbox"
                        checked={selectedFiles.has(file.path)}
                        onChange={() => toggleFile(file.path)}
                        className="w-4 h-4 mt-1"
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-1">
                          <h3 className="font-medium text-gray-900">{file.name}</h3>
                          {getStatusBadge(file.status)}
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <div>Size: {formatFileSize(file.size)}</div>
                          {file.date && <div>Date: {file.date}</div>}

                          {/* Show both paths and sizes for duplicates */}
                          {file.status === 'duplicate' && file.existing_path && (
                            <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                              <div className="font-semibold text-yellow-900 mb-1">📋 Duplicate Comparison:</div>
                              <div className="space-y-1">
                                <div>
                                  <span className="text-gray-500">Source (!NAGRANIA):</span><br/>
                                  <span className="text-gray-700">{file.path}</span><br/>
                                  <span className="font-medium">Size: {formatFileSize(file.size)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Existing (SORTED):</span><br/>
                                  <span className="text-gray-700">{file.existing_path}</span><br/>
                                  <span className="font-medium">Size: {formatFileSize(file.existing_size || 0)}</span>
                                </div>
                                {file.size === file.existing_size && (
                                  <div className="text-green-700 font-medium">✓ Sizes match - safe to delete</div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Show both paths and sizes for files with different sizes */}
                          {file.status === 'exists_different_size' && file.existing_path && (
                            <div className="mt-2 p-2 bg-orange-50 border border-orange-200 rounded text-xs">
                              <div className="font-semibold text-orange-900 mb-1">⚠️ Different Size Warning:</div>
                              <div className="space-y-1">
                                <div>
                                  <span className="text-gray-500">Source (!NAGRANIA):</span><br/>
                                  <span className="text-gray-700">{file.path}</span><br/>
                                  <span className="font-medium">Size: {formatFileSize(file.size)}</span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Existing (SORTED):</span><br/>
                                  <span className="text-gray-700">{file.existing_path}</span><br/>
                                  <span className="font-medium">Size: {formatFileSize(file.existing_size || 0)}</span>
                                </div>
                                <div className="text-orange-700 font-medium">
                                  ⚠️ Sizes differ by {Math.abs(file.size - (file.existing_size || 0)) / (1024 * 1024).toFixed(1)} MB
                                </div>
                              </div>
                            </div>
                          )}

                          {file.target_path && file.status !== 'duplicate' && file.status !== 'exists_different_size' && (
                            <div className="text-xs text-gray-500">→ {file.target_path}</div>
                          )}
                          {file.error && (
                            <div className="text-xs text-red-600">⚠️ {file.error}</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {files.length === 0 && !scanning && (
          <div className="text-center py-12 text-gray-500">
            Click "Scan for New Files" to start
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

      {/* Sort Confirmation */}
      <Toast
        show={sortConfirm.show}
        onClose={() => setSortConfirm({ show: false, count: 0 })}
        title="Sort Files?"
        message={`Sort ${sortConfirm.count} file${sortConfirm.count !== 1 ? 's' : ''}? They will be moved to SORTED folder.`}
        icon="📁"
        color="blue"
        index={(successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0)}
        autoClose={0}
        actions={[
          {
            label: 'Sort',
            onClick: performSort,
            color: 'primary'
          },
          {
            label: 'Cancel',
            onClick: () => setSortConfirm({ show: false, count: 0 }),
            color: 'secondary'
          }
        ]}
      />

      {/* Sort Success - with Analyze option */}
      <Toast
        show={sortSuccess.show}
        onClose={() => setSortSuccess({ show: false, moved: 0, duplicates: 0, renamed: 0, errors: 0, movedPaths: [] })}
        title="Sort Complete!"
        message={`Moved: ${sortSuccess.moved} • Duplicates removed: ${sortSuccess.duplicates} • Renamed: ${sortSuccess.renamed} • Errors: ${sortSuccess.errors}`}
        icon="✅"
        color="green"
        index={(successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0) + (sortConfirm.show ? 1 : 0)}
        autoClose={0}
        actions={sortSuccess.movedPaths.length > 0 ? [
          {
            label: 'Analyze Now',
            onClick: () => handleBatchAnalyze(sortSuccess.movedPaths),
            color: 'primary'
          },
          {
            label: 'Skip',
            onClick: () => setSortSuccess({ show: false, moved: 0, duplicates: 0, renamed: 0, errors: 0, movedPaths: [] }),
            color: 'secondary'
          }
        ] : undefined}
      />

      {/* Delete Duplicates Confirmation */}
      <Toast
        show={deleteDuplicatesConfirm.show}
        onClose={() => setDeleteDuplicatesConfirm({ show: false, count: 0 })}
        title="Delete Duplicates?"
        message={`Delete ${deleteDuplicatesConfirm.count} duplicate file${deleteDuplicatesConfirm.count !== 1 ? 's' : ''} from !NAGRANIA KONCERTÓW? Files in SORTED will be kept.`}
        icon="⚠️"
        color="red"
        index={(successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0) + (sortConfirm.show ? 1 : 0) + (sortSuccess.show ? 1 : 0)}
        autoClose={0}
        actions={[
          {
            label: 'Delete',
            onClick: performDeleteDuplicates,
            color: 'danger'
          },
          {
            label: 'Cancel',
            onClick: () => setDeleteDuplicatesConfirm({ show: false, count: 0 }),
            color: 'secondary'
          }
        ]}
      />
    </div>
  )
}
