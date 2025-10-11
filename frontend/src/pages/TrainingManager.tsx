import { useState, useEffect } from 'react'
import axios from 'axios'
import Toast from '../components/Toast'

interface TrainingManagerProps {
  onBack: () => void
}

interface TrainingDataStats {
  stats_per_class: Record<string, {
    count: number
    duration_sec: number
    duration_min: number
  }>
  total_duration_sec: number
  total_duration_min: number
  total_count: number
}

interface TrainingStatus {
  job_id: string
  status: 'preparing' | 'training' | 'completed' | 'failed' | 'cancelled'
  current_epoch: number
  total_epochs: number
  training_acc: number
  training_loss: number
  val_acc: number
  val_loss: number
  progress: number
  time_elapsed_sec: number
  time_remaining_sec: number | null
  samples_per_class: Record<string, number>
  error: string | null
  model_filename: string | null
}

interface ModelInfo {
  filename: string
  model_id: string
  trained_date: string
  accuracy: number
  val_accuracy: number
  loss: number
  val_loss: number
  epochs_trained: number
  training_samples: number
  notes: string
  per_class_acc?: Record<string, number>
  measured_train_acc?: number
  test_accuracy?: number
  dataset_measured_on?: string
}

interface ModelsResponse {
  active_model: string
  models: ModelInfo[]
}

export default function TrainingManager({ onBack }: TrainingManagerProps) {
  const [dataStats, setDataStats] = useState<TrainingDataStats | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [showModels, setShowModels] = useState(false)

  const [successToast, setSuccessToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [errorToast, setErrorToast] = useState<{ show: boolean; message: string }>({ show: false, message: '' })
  const [confirmToast, setConfirmToast] = useState<{ show: boolean; message: string; action: () => void }>({
    show: false,
    message: '',
    action: () => {}
  })
  const [measuringModel, setMeasuringModel] = useState<string | null>(null)

  // Load data stats, models, and check for active training on mount
  useEffect(() => {
    loadDataStats()
    loadModels()
    checkForActiveJob()
  }, [])

  const checkForActiveJob = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/training/active-job')
      if (response.data.job_id) {
        setActiveJobId(response.data.job_id)
        setTrainingStatus(response.data)
        setAutoRefresh(true)
      }
    } catch (error: any) {
      console.error('Error checking for active job:', error)
    }
  }

  // Auto-refresh training status
  useEffect(() => {
    if (!autoRefresh || !activeJobId) return

    const interval = setInterval(async () => {
      await fetchTrainingStatus(activeJobId)
    }, 1000)  // Update every second

    return () => clearInterval(interval)
  }, [autoRefresh, activeJobId])

  const loadDataStats = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/training/data-stats')
      setDataStats(response.data)
    } catch (error: any) {
      console.error('Error loading data stats:', error)
      setErrorToast({ show: true, message: 'Failed to load training data stats' })
    }
  }

  const loadModels = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/training/models')
      setModels(response.data)
    } catch (error: any) {
      console.error('Error loading models:', error)
    }
  }

  const fetchTrainingStatus = async (jobId: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/v1/training/status/${jobId}`)
      setTrainingStatus(response.data)

      // Stop auto-refresh when completed/failed/cancelled
      if (['completed', 'failed', 'cancelled'].includes(response.data.status)) {
        setAutoRefresh(false)
        if (response.data.status === 'completed') {
          setSuccessToast({ show: true, message: `Training completed! Model saved: ${response.data.model_filename}` })
          loadModels()  // Reload models list
        } else if (response.data.status === 'failed') {
          setErrorToast({ show: true, message: `Training failed: ${response.data.error}` })
        }
      }
    } catch (error: any) {
      console.error('Error fetching training status:', error)
    }
  }

  const handleStartTraining = async () => {
    if (!dataStats || dataStats.total_count === 0) {
      setErrorToast({ show: true, message: 'No training data found' })
      return
    }

    try {
      const response = await axios.post('http://localhost:8000/api/v1/training/start')
      const jobId = response.data.job_id
      setActiveJobId(jobId)
      setAutoRefresh(true)
      setSuccessToast({ show: true, message: 'Training started!' })
      fetchTrainingStatus(jobId)
    } catch (error: any) {
      console.error('Error starting training:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to start training' })
    }
  }

  const handleCancelTraining = async () => {
    if (!activeJobId) return

    try {
      await axios.post(`http://localhost:8000/api/v1/training/${activeJobId}/cancel`)
      setSuccessToast({ show: true, message: 'Training cancellation requested' })
    } catch (error: any) {
      console.error('Error cancelling training:', error)
      setErrorToast({ show: true, message: 'Failed to cancel training' })
    }
  }

  const handleActivateModel = async (filename: string) => {
    try {
      setSuccessToast({ show: true, message: 'Activating and loading model...' })
      const response = await axios.post('http://localhost:8000/api/v1/training/activate-model', { filename })

      setSuccessToast({
        show: true,
        message: `✅ ${filename} activated and loaded!\n🎯 Model is now ready for analysis.`
      })

      loadModels()  // Reload to show new active status
      setConfirmToast({ show: false, message: '', action: () => {} })
    } catch (error: any) {
      console.error('Error activating model:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to activate model' })
    }
  }

  const handleDeleteModel = async (filename: string) => {
    try {
      await axios.delete(`http://localhost:8000/api/v1/training/models/${filename}`)
      setSuccessToast({ show: true, message: `Model ${filename} deleted` })
      loadModels()
      setConfirmToast({ show: false, message: '', action: () => {} })
    } catch (error: any) {
      console.error('Error deleting model:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to delete model' })
    }
  }

  const handleMeasureAccuracy = async (filename: string) => {
    try {
      setMeasuringModel(filename)
      setSuccessToast({ show: true, message: 'Measuring accuracy... This may take 1-2 minutes.' })
      const response = await axios.post(`http://localhost:8000/api/v1/training/measure-accuracy/${filename}`)
      const { train_acc, val_acc, test_acc, per_class_acc, dataset_used } = response.data

      // Build per-class summary
      const perClassStr = per_class_acc
        ? Object.entries(per_class_acc)
            .map(([cls, acc]: [string, any]) => `${cls}: ${acc.toFixed(1)}%`)
            .join(', ')
        : 'N/A'

      setSuccessToast({
        show: true,
        message: `✅ Measured on ${dataset_used || 'dataset'}:\nTrain ${train_acc.toFixed(2)}%, Val ${val_acc.toFixed(2)}%, Test ${test_acc.toFixed(2)}%\n${perClassStr}`
      })
      loadModels()  // Reload to show updated metadata
    } catch (error: any) {
      console.error('Error measuring accuracy:', error)
      setErrorToast({ show: true, message: error.response?.data?.detail || 'Failed to measure accuracy' })
    } finally {
      setMeasuringModel(null)
    }
  }

  const formatTime = (seconds: number | null) => {
    if (seconds === null || seconds === 0) return '--'
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const formatDate = (isoDate: string) => {
    const date = new Date(isoDate)
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getStatusBadge = (status: string) => {
    const badges: Record<string, { color: string; text: string }> = {
      preparing: { color: 'bg-yellow-100 text-yellow-800', text: 'Preparing...' },
      training: { color: 'bg-blue-100 text-blue-800', text: 'Training' },
      completed: { color: 'bg-green-100 text-green-800', text: 'Completed' },
      failed: { color: 'bg-red-100 text-red-800', text: 'Failed' },
      cancelled: { color: 'bg-orange-100 text-orange-800', text: 'Cancelled' }
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
          <h1 className="text-3xl font-bold">🎓 Model Training</h1>
          <p className="text-gray-600 text-sm mt-1">
            Retrain the AI model with new labeled audio segments
          </p>
        </div>

        {/* Training Data Summary */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">📊 Training Data Summary</h2>

          {dataStats ? (
            <>
              <div className="space-y-2 mb-4">
                {Object.entries(dataStats.stats_per_class).map(([className, stats]) => {
                  const durationMin = stats.duration_min
                  const isLow = durationMin < 2.5  // Less than 2.5 minutes
                  return (
                    <div key={className} className="flex items-center justify-between">
                      <span className="font-medium">{className}:</span>
                      <span className={`${isLow ? 'text-orange-600' : 'text-gray-700'}`}>
                        {durationMin.toFixed(1)} min ({stats.count} files) {isLow && '⚠️ LOW'}
                      </span>
                    </div>
                  )
                })}
                <div className="border-t pt-2 mt-2 flex items-center justify-between font-semibold">
                  <span>Total:</span>
                  <span>{dataStats.total_duration_min.toFixed(1)} min ({dataStats.total_count} files)</span>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleStartTraining}
                  disabled={!!trainingStatus && trainingStatus.status === 'training'}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  🚀 Start Training
                </button>
                <button
                  onClick={() => setShowModels(!showModels)}
                  className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium"
                >
                  📚 {showModels ? 'Hide' : 'View'} Models
                </button>
              </div>
            </>
          ) : (
            <p className="text-gray-500">Loading training data stats...</p>
          )}
        </div>

        {/* Current Training Job */}
        {trainingStatus && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">🔄 Current Training Job</h2>
              {getStatusBadge(trainingStatus.status)}
            </div>

            {trainingStatus.status === 'preparing' && (
              <p className="text-gray-600">Loading training data and preparing model...</p>
            )}

            {trainingStatus.status === 'training' && (
              <>
                <div className="mb-4">
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="font-medium">
                      Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs}
                    </span>
                    <span className="text-gray-600">{trainingStatus.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-blue-600 h-3 rounded-full transition-all"
                      style={{ width: `${trainingStatus.progress}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                  <div>
                    <p className="text-gray-600">Training:</p>
                    <p className="font-mono">
                      acc={trainingStatus.training_acc.toFixed(4)} loss={trainingStatus.training_loss.toFixed(4)}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Validation:</p>
                    <p className="font-mono">
                      acc={trainingStatus.val_acc.toFixed(4)} loss={trainingStatus.val_loss.toFixed(4)}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm mb-4">
                  <span className="text-gray-600">
                    Time elapsed: <span className="font-medium">{formatTime(trainingStatus.time_elapsed_sec)}</span>
                  </span>
                  <span className="text-gray-600">
                    Est. remaining: <span className="font-medium">{formatTime(trainingStatus.time_remaining_sec)}</span>
                  </span>
                </div>

                <button
                  onClick={handleCancelTraining}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm font-medium"
                >
                  ✕ Cancel Training
                </button>
              </>
            )}

            {trainingStatus.status === 'completed' && (
              <div className="text-green-700">
                <p className="font-medium mb-2">✅ Training completed successfully!</p>
                <p className="text-sm">Model saved: <span className="font-mono">{trainingStatus.model_filename}</span></p>
                <p className="text-sm">Final validation accuracy: {(trainingStatus.val_acc * 100).toFixed(2)}%</p>
                <p className="text-sm">Epochs trained: {trainingStatus.current_epoch}</p>
              </div>
            )}

            {trainingStatus.status === 'failed' && (
              <div className="text-red-700">
                <p className="font-medium mb-2">❌ Training failed</p>
                <p className="text-sm font-mono bg-red-50 p-2 rounded">{trainingStatus.error}</p>
              </div>
            )}

            {trainingStatus.status === 'cancelled' && (
              <p className="text-orange-700">⚠️ Training was cancelled</p>
            )}
          </div>
        )}

        {/* Available Models */}
        {showModels && models && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">📚 Available Models</h2>

            {models.models.length === 0 ? (
              <p className="text-gray-500">No trained models found</p>
            ) : (
              <div className="space-y-4">
                {models.models.map((model) => {
                  const isActive = model.model_id === models.active_model
                  return (
                    <div
                      key={model.filename}
                      className={`border rounded-lg p-4 ${isActive ? 'border-green-500 bg-green-50' : 'border-gray-200'}`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className={`text-2xl ${isActive ? 'text-green-600' : 'text-gray-400'}`}>
                            {isActive ? '●' : '○'}
                          </span>
                          <div>
                            <p className="font-mono text-sm font-semibold">{model.filename}</p>
                            {isActive && (
                              <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded">ACTIVE</span>
                            )}
                          </div>
                        </div>
                        <p className="text-xs text-gray-500">{formatDate(model.trained_date)}</p>
                      </div>

                      {/* Main metrics */}
                      <div className="grid grid-cols-4 gap-2 text-sm mb-3">
                        <div>
                          <p className="text-gray-600" title="Measured accuracy on training set (higher = better)">
                            Train Acc
                          </p>
                          <p className="font-semibold">
                            {model.measured_train_acc
                              ? `${(model.measured_train_acc * 100).toFixed(2)}%`
                              : `${(model.accuracy * 100).toFixed(2)}%`}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-600" title="Measured accuracy on validation set (higher = better)">
                            Val Acc
                          </p>
                          <p className="font-semibold">{(model.val_accuracy * 100).toFixed(2)}%</p>
                        </div>
                        <div>
                          <p className="text-gray-600" title="Measured accuracy on test set (higher = better)">
                            Test Acc
                          </p>
                          <p className="font-semibold">
                            {model.test_accuracy
                              ? `${(model.test_accuracy * 100).toFixed(2)}%`
                              : '--'}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-600">Epochs</p>
                          <p className="font-semibold">{model.epochs_trained}</p>
                        </div>
                      </div>

                      {/* Per-class Accuracy - one line */}
                      {model.per_class_acc && Object.keys(model.per_class_acc).length > 0 && (
                        <div className="bg-blue-50 border border-blue-200 rounded p-2 mb-3">
                          <div className="flex items-center justify-between gap-2 text-xs">
                            <span className="font-semibold text-gray-700 whitespace-nowrap">Per-class:</span>
                            <div className="flex gap-3 flex-wrap">
                              {Object.entries(model.per_class_acc).map(([cls, acc]: [string, any]) => (
                                <span key={cls} className="whitespace-nowrap">
                                  <span className="text-gray-600">{cls}</span>{' '}
                                  <span className="font-semibold">{(acc * 100).toFixed(1)}%</span>
                                </span>
                              ))}
                            </div>
                          </div>
                          {model.dataset_measured_on && (
                            <p className="text-xs text-gray-500 mt-1">
                              Measured on: {model.dataset_measured_on}
                            </p>
                          )}
                        </div>
                      )}

                      {model.notes && (
                        <p className="text-xs text-gray-600 mb-3">{model.notes}</p>
                      )}

                      <div className="flex gap-2">
                        {isActive && (
                          <button
                            onClick={() => handleMeasureAccuracy(model.filename)}
                            disabled={measuringModel === model.filename}
                            className={`px-3 py-1 ${measuringModel === model.filename ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded text-sm`}
                            title="Measure true train/val/test accuracy on natural data"
                          >
                            {measuringModel === model.filename ? '🔄 Measuring...' : '📊 Measure'}
                          </button>
                        )}

                        {!isActive && (
                          <>
                          <button
                            onClick={() => {
                              setConfirmToast({
                                show: true,
                                message: `Activate ${model.filename}? Requires backend restart.`,
                                action: () => handleActivateModel(model.filename)
                              })
                            }}
                            className="px-3 py-1 bg-green-600 text-white rounded text-sm hover:bg-green-700"
                          >
                            Activate
                          </button>
                          <button
                            onClick={() => handleMeasureAccuracy(model.filename)}
                            disabled={measuringModel === model.filename}
                            className={`px-3 py-1 ${measuringModel === model.filename ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} text-white rounded text-sm`}
                            title="Measure true train/val/test accuracy on natural data"
                          >
                            {measuringModel === model.filename ? '🔄 Measuring...' : '📊 Measure'}
                          </button>
                          <button
                            onClick={() => {
                              setConfirmToast({
                                show: true,
                                message: `Delete ${model.filename}? This cannot be undone.`,
                                action: () => handleDeleteModel(model.filename)
                              })
                            }}
                            className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
                          >
                            🗑️ Delete
                          </button>
                          </>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* Toasts */}
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

        <Toast
          show={errorToast.show}
          onClose={() => setErrorToast({ show: false, message: '' })}
          title="Error"
          message={errorToast.message}
          icon="❌"
          color="red"
          index={successToast.show ? 1 : 0}
          autoClose={6000}
        />

        <Toast
          show={confirmToast.show}
          onClose={() => setConfirmToast({ show: false, message: '', action: () => {} })}
          title="Confirm Action"
          message={confirmToast.message}
          icon="⚠️"
          color="yellow"
          index={(successToast.show ? 1 : 0) + (errorToast.show ? 1 : 0)}
          autoClose={0}
          actions={[
            { label: 'Confirm', onClick: confirmToast.action, color: 'primary' },
            { label: 'Cancel', onClick: () => setConfirmToast({ show: false, message: '', action: () => {} }), color: 'secondary' }
          ]}
        />
      </div>
    </div>
  )
}
