import { useToastStore } from '../stores/toastStore'
import Toast from './Toast'

export default function ToastContainer() {
  const toasts = useToastStore((state) => state.toasts)
  const removeToast = useToastStore((state) => state.removeToast)

  return (
    <>
      {toasts.map((toast, index) => (
        <Toast
          key={toast.id}
          show={true}
          title={toast.title}
          message={toast.message}
          color={toast.color}
          icon={toast.icon}
          index={index}
          onClose={() => removeToast(toast.id)}
          // Error toasts (red) require manual dismiss - no autoClose
          // Success toasts (green) auto-dismiss after 5 seconds
          autoClose={toast.color === 'green' ? 5000 : undefined}
          actions={toast.retry ? [{
            label: 'Retry',
            onClick: toast.retry,
            color: 'primary'
          }] : undefined}
        />
      ))}
    </>
  )
}
