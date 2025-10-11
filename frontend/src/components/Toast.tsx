import { useEffect } from 'react'

interface ToastAction {
  label: string
  onClick: () => void
  color?: 'primary' | 'secondary' | 'danger'
}

interface ToastProps {
  show: boolean
  onClose: () => void
  title: string
  message?: string
  icon?: string
  color?: 'green' | 'purple' | 'blue' | 'red' | 'yellow'
  actions?: ToastAction[]
  index?: number  // For stacking multiple toasts
  autoClose?: number  // Auto-close timeout in ms (0 = no auto-close)
}

const colorClasses = {
  green: {
    bg: 'bg-green-600',
    hover: 'hover:bg-green-700',
    text: 'text-green-100'
  },
  purple: {
    bg: 'bg-purple-600',
    hover: 'hover:bg-purple-700',
    text: 'text-purple-100'
  },
  blue: {
    bg: 'bg-blue-600',
    hover: 'hover:bg-blue-700',
    text: 'text-blue-100'
  },
  red: {
    bg: 'bg-red-600',
    hover: 'hover:bg-red-700',
    text: 'text-red-100'
  },
  yellow: {
    bg: 'bg-yellow-600',
    hover: 'hover:bg-yellow-700',
    text: 'text-yellow-100'
  }
}

const actionColorClasses = {
  primary: 'bg-white text-gray-800 hover:bg-gray-100',
  secondary: 'bg-transparent border border-white text-white hover:bg-white hover:bg-opacity-20',
  danger: 'bg-red-700 text-white hover:bg-red-800'
}

export default function Toast({
  show,
  onClose,
  title,
  message,
  icon = '✓',
  color = 'green',
  actions,
  index = 0,
  autoClose
}: ToastProps) {
  const colors = colorClasses[color]
  const topOffset = 16 + index * 100  // Stack toasts 100px apart

  // Auto-close timer
  useEffect(() => {
    if (show && autoClose && autoClose > 0) {
      const timer = setTimeout(() => onClose(), autoClose)
      return () => clearTimeout(timer)
    }
  }, [show, autoClose, onClose])

  if (!show) return null

  return (
    <div
      className="fixed right-4 z-50 animate-fade-in"
      style={{ top: `${topOffset}px` }}
    >
      <div
        className={`${colors.bg} text-white rounded-lg shadow-lg px-4 py-3 flex items-center gap-3 transition-colors ${!actions ? 'cursor-pointer ' + colors.hover : ''}`}
        onClick={!actions ? onClose : undefined}
      >
        <div className="text-2xl">{icon}</div>
        <div className="flex-1">
          <h3 className="font-semibold">{title}</h3>
          {message && (
            <p className={`text-sm ${colors.text}`}>{message}</p>
          )}

          {/* Action buttons (for Yes/No, OK/Cancel, etc.) */}
          {actions && actions.length > 0 && (
            <div className="flex gap-2 mt-3">
              {actions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={action.onClick}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${actionColorClasses[action.color || 'secondary']}`}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}

          {/* Dismiss hint (only for simple toasts without actions) */}
          {!actions && (
            <p className={`text-xs ${colors.text} mt-1`}>Click to dismiss</p>
          )}
        </div>
      </div>
    </div>
  )
}
