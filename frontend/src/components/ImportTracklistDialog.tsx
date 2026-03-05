import { useState, useEffect } from 'react'
import axios from 'axios'

interface ProgramPiece {
  composer: string
  title: string
  duration_min: number | null
  annotation: string | null
  is_break: boolean
}

interface ConcertProgram {
  title: string
  date: string
  time: string
  venue: string
  conductor: string | null
  soloists: string[]
  orchestra: string | null
  pieces: ProgramPiece[]
  url: string
}

interface Track {
  id: string
  predicted_class: string
  name: string
  start: string
  stop: string
  duration: string
  selected: boolean
  [key: string]: any
}

interface ImportTracklistDialogProps {
  show: boolean
  onClose: () => void
  csvName: string | null
  tracks: Track[]
  onImport: (updatedTracks: Track[]) => void
}

export function ImportTracklistDialog({ show, onClose, csvName, tracks, onImport }: ImportTracklistDialogProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [concerts, setConcerts] = useState<ConcertProgram[]>([])
  const [selectedConcert, setSelectedConcert] = useState<ConcertProgram | null>(null)
  const [step, setStep] = useState<'loading' | 'select-concert' | 'preview' | 'manual'>('loading')
  const [selectedPieces, setSelectedPieces] = useState<Set<number>>(new Set())

  // Extract date from CSV name: predictions_SONG042_2026-03-06_19-30.csv
  const extractDate = (name: string | null): string | null => {
    if (!name) return null
    const match = name.match(/(\d{4}-\d{2}-\d{2})/)
    return match ? match[1] : null
  }

  useEffect(() => {
    if (!show) return
    setError(null)
    setConcerts([])
    setSelectedConcert(null)

    const date = extractDate(csvName)
    if (!date) {
      setError('Nie udało się wyciągnąć daty z nazwy pliku CSV')
      setStep('loading')
      return
    }

    setLoading(true)
    setStep('loading')

    axios.get(`/api/v1/filharmonia/concerts?date=${date}`)
      .then(res => {
        const data: ConcertProgram[] = res.data
        setConcerts(data)
        if (data.length === 0) {
          setError(`Nie znaleziono koncertów na ${date}`)
          setStep('loading')
        } else if (data.length === 1) {
          setSelectedConcert(data[0])
          // Auto-select all non-break pieces
          const nonBreak = data[0].pieces.reduce((acc: number[], p, i) => {
            if (!p.is_break) acc.push(i)
            return acc
          }, [])
          setSelectedPieces(new Set(nonBreak))
          setStep('preview')
        } else {
          setStep('select-concert')
        }
      })
      .catch(err => {
        setError(err.response?.data?.detail || 'Błąd pobierania danych z filharmonia.pl')
        setStep('loading')
      })
      .finally(() => setLoading(false))
  }, [show, csvName])

  const musicTracks = tracks.filter(t => t.predicted_class === 'MUSIC')
  const allPieces = selectedConcert?.pieces || []
  // Only selected non-break pieces for import
  const piecesToImport = allPieces.filter((p, i) => selectedPieces.has(i) && !p.is_break)

  const togglePiece = (index: number) => {
    setSelectedPieces(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const selectHalf = (half: 'first' | 'second') => {
    // Find break index to split into halves
    const breakIndex = allPieces.findIndex(p => p.is_break)
    const newSet = new Set<number>()
    if (breakIndex === -1) {
      // No break - select all
      allPieces.forEach((p, i) => { if (!p.is_break) newSet.add(i) })
    } else if (half === 'first') {
      allPieces.forEach((p, i) => { if (i < breakIndex && !p.is_break) newSet.add(i) })
    } else {
      allPieces.forEach((p, i) => { if (i > breakIndex && !p.is_break) newSet.add(i) })
    }
    setSelectedPieces(newSet)
  }

  const handleImport = () => {
    if (!selectedConcert || piecesToImport.length === 0) return

    const updatedTracks = [...tracks]

    // Map selected pieces sequentially to MUSIC tracks
    let pieceIndex = 0
    for (let i = 0; i < updatedTracks.length && pieceIndex < piecesToImport.length; i++) {
      if (updatedTracks[i].predicted_class === 'MUSIC') {
        const piece = piecesToImport[pieceIndex]
        updatedTracks[i] = {
          ...updatedTracks[i],
          name: `${piece.composer} - ${piece.title}${piece.duration_min ? ` [${piece.duration_min}']` : ''}`
        }
        pieceIndex++
      }
    }

    onImport(updatedTracks)
    onClose()
  }

  if (!show) return null

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Import programu z filharmonia.pl
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl">&times;</button>
        </div>

        {/* Loading */}
        {loading && (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-3"></div>
            <p className="text-gray-600">Szukam koncertów na filharmonia.pl...</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="bg-red-50 border border-red-200 rounded p-4 mb-4">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Select Concert (multiple results) */}
        {step === 'select-concert' && !loading && (
          <div>
            <p className="text-sm text-gray-600 mb-3">Znaleziono {concerts.length} koncerty. Wybierz:</p>
            <div className="space-y-2">
              {concerts.map((concert, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setSelectedConcert(concert)
                    const nonBreak = concert.pieces.reduce((acc: number[], p, i) => {
                      if (!p.is_break) acc.push(i)
                      return acc
                    }, [])
                    setSelectedPieces(new Set(nonBreak))
                    setStep('preview')
                  }}
                  className="w-full text-left p-3 border rounded hover:bg-blue-50 hover:border-blue-300 transition-colors"
                >
                  <div className="font-medium">{concert.title}</div>
                  <div className="text-sm text-gray-500">
                    {concert.time} &bull; {concert.venue}
                    {concert.conductor && ` &bull; dyr. ${concert.conductor}`}
                  </div>
                  <div className="text-sm text-gray-400">
                    {concert.pieces.filter(p => !p.is_break).length} utworów
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Preview mapping */}
        {step === 'preview' && selectedConcert && !loading && (
          <div>
            <div className="mb-4 p-3 bg-gray-50 rounded">
              <div className="font-medium">{selectedConcert.title}</div>
              <div className="text-sm text-gray-600">
                {selectedConcert.date} {selectedConcert.time} &bull; {selectedConcert.venue}
              </div>
              {selectedConcert.conductor && (
                <div className="text-sm text-gray-500">Dyrygent: {selectedConcert.conductor}</div>
              )}
              {selectedConcert.orchestra && (
                <div className="text-sm text-gray-500">{selectedConcert.orchestra}</div>
              )}
              {concerts.length > 1 && (
                <button
                  onClick={() => setStep('select-concert')}
                  className="text-sm text-blue-600 hover:underline mt-1"
                >
                  Zmień koncert
                </button>
              )}
            </div>

            {/* Quick select buttons */}
            {allPieces.some(p => p.is_break) && (
              <div className="flex gap-2 mb-3">
                <span className="text-sm text-gray-500 self-center">Wybierz:</span>
                <button
                  onClick={() => selectHalf('first')}
                  className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200 border"
                >
                  I połowa
                </button>
                <button
                  onClick={() => selectHalf('second')}
                  className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200 border"
                >
                  II połowa
                </button>
                <button
                  onClick={() => {
                    const all = new Set<number>()
                    allPieces.forEach((p, i) => { if (!p.is_break) all.add(i) })
                    setSelectedPieces(all)
                  }}
                  className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200 border"
                >
                  Wszystko
                </button>
              </div>
            )}

            {/* Program with checkboxes */}
            <div className="border rounded overflow-hidden mb-3">
              <div className="px-3 py-2 bg-gray-100 border-b font-medium text-sm">
                Program ({allPieces.filter(p => !p.is_break).length} utworów)
              </div>
              <div className="max-h-[35vh] overflow-y-auto">
                {allPieces.map((piece, i) => {
                  if (piece.is_break) {
                    return (
                      <div key={i} className="px-3 py-2 text-sm text-gray-400 bg-gray-50 border-b border-dashed text-center italic">
                        --- przerwa {piece.duration_min ? `[${piece.duration_min}']` : ''} ---
                      </div>
                    )
                  }
                  const isSelected = selectedPieces.has(i)
                  return (
                    <label
                      key={i}
                      className={`flex items-center gap-3 px-3 py-2 border-b last:border-0 cursor-pointer hover:bg-blue-50 ${isSelected ? 'bg-green-50' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => togglePiece(i)}
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <div className="text-sm flex-1">
                        <span className="font-medium">{piece.composer}</span>
                        {piece.composer && ' - '}
                        <span>{piece.title}</span>
                        {piece.duration_min && (
                          <span className="text-gray-400 ml-1">[{piece.duration_min}']</span>
                        )}
                        {piece.annotation && (
                          <span className="text-gray-400 ml-1 italic">({piece.annotation})</span>
                        )}
                      </div>
                    </label>
                  )
                })}
              </div>
            </div>

            {/* Mapping preview */}
            {piecesToImport.length > 0 && (
              <div className="bg-blue-50 border border-blue-200 rounded p-3 mb-3 text-sm">
                <span className="font-medium">{piecesToImport.length}</span> wybranych utworów
                {' → '}
                <span className="font-medium">{musicTracks.length}</span> segmentów MUSIC
                {piecesToImport.length !== musicTracks.length && (
                  <span className="text-yellow-700 ml-1">(przypisane po kolei)</span>
                )}
              </div>
            )}

            <div className="flex gap-2 justify-end">
              <button
                onClick={onClose}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
              >
                Anuluj
              </button>
              <button
                onClick={handleImport}
                disabled={piecesToImport.length === 0}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300"
              >
                Importuj ({Math.min(musicTracks.length, piecesToImport.length)} nazw)
              </button>
            </div>
          </div>
        )}

        {/* No results + error state - close button */}
        {error && !loading && (
          <div className="flex justify-end mt-4">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
            >
              Zamknij
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
