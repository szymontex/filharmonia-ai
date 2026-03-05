import { useEffect, useState } from 'react'
import axios from 'axios'
import CsvViewer from './pages/CsvViewer'
import CalendarBrowser from './pages/CalendarBrowser'
import SortManager from './pages/SortManager'
import AnalysisMonitor from './pages/AnalysisMonitor'
import TrainingManager from './pages/TrainingManager'
import UncertaintyReview from './pages/UncertaintyReview'
import ToastContainer from './components/ToastContainer'

function App() {
  const [page, setPage] = useState<'home' | 'csv' | 'calendar' | 'sort' | 'monitor' | 'training' | 'uncertainty'>('home')
  const [csvToOpen, setCsvToOpen] = useState<string | null>(null)

  return (
    <>
      <ToastContainer />
      {page === 'csv' && <CsvViewer onBack={() => setPage('home')} initialCsv={csvToOpen} />}
      {page === 'calendar' && (
        <CalendarBrowser
          onBack={() => setPage('home')}
          onOpenCsv={(csvPath) => {
            setCsvToOpen(csvPath)
            setPage('csv')
          }}
        />
      )}
      {page === 'sort' && <SortManager onBack={() => setPage('home')} />}
      {page === 'monitor' && <AnalysisMonitor onBack={() => setPage('home')} />}
      {page === 'training' && <TrainingManager onBack={() => setPage('home')} />}
      {page === 'uncertainty' && <UncertaintyReview onBack={() => setPage('home')} />}
      {page === 'home' && <HomePage onNavigate={setPage} />}
    </>
  )
}

function HomePage({ onNavigate }: { onNavigate: (page: 'home' | 'csv' | 'calendar' | 'sort' | 'monitor' | 'training' | 'uncertainty') => void }) {
  const [status, setStatus] = useState<string>('Checking backend...')
  const [gpuAvailable, setGpuAvailable] = useState<boolean>(false)

  useEffect(() => {
    axios.get('/health')
      .then(res => {
        setStatus(`✅ Backend: ${res.data.status}`)
        // /health returns device: "cuda" or "cpu"
        setGpuAvailable(res.data.device === 'cuda')
      })
      .catch(() => setStatus('❌ Backend offline'))
  }, [])

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-4 text-gray-900">
          🎵 Filharmonia AI
        </h1>

        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <h2 className="text-xl font-semibold mb-2">System Status</h2>
          <p className="text-lg mb-2">{status}</p>
          <p className="text-sm text-gray-600">
            GPU: {gpuAvailable ? '✅ Available' : '⚠️ Not detected (CPU mode)'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-4">
          <h2 className="text-xl font-semibold mb-2">Quick Start</h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>Upload MP3 files from concerts</li>
            <li>Click "Sort" to organize by date</li>
            <li>Click "Analyze" to run AI classification</li>
            <li>Edit markers in waveform editor</li>
            <li>Export tracklist</li>
          </ol>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-2">Tools</h2>
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => onNavigate('sort')}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium"
            >
              🗂️ Sort New Recordings
            </button>
            <button
              onClick={() => onNavigate('calendar')}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
            >
              📅 Browse Recordings
            </button>
            <button
              onClick={() => onNavigate('monitor')}
              className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-medium"
            >
              📊 Analysis Monitor
            </button>
            <button
              onClick={() => onNavigate('csv')}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
            >
              📝 CSV Track Editor
            </button>
            <button
              onClick={() => onNavigate('uncertainty')}
              className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 font-medium"
            >
              🎲 Uncertainty Review
            </button>
            <button
              onClick={() => onNavigate('training')}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium"
            >
              🎓 Model Training
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
