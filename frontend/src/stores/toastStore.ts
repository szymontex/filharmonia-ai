import { create } from 'zustand'

interface ToastItem {
  id: string
  title: string
  message?: string
  color: 'green' | 'red' | 'blue' | 'yellow' | 'purple'
  icon?: string
  retry?: () => void
}

interface ToastStore {
  toasts: ToastItem[]
  addToast: (toast: Omit<ToastItem, 'id'>) => void
  removeToast: (id: string) => void
  clearAll: () => void
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],

  addToast: (toast) => set((state) => {
    // Generate UUID-style id
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2)
    const newToast = { ...toast, id }

    // Cap at 5 toasts - remove oldest if exceeded
    const updatedToasts = [...state.toasts, newToast]
    if (updatedToasts.length > 5) {
      updatedToasts.shift() // Remove oldest
    }

    return { toasts: updatedToasts }
  }),

  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter(toast => toast.id !== id)
  })),

  clearAll: () => set({ toasts: [] })
}))
