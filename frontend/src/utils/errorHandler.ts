import axios, { AxiosError } from 'axios'
import { useToastStore } from '../stores/toastStore'

/**
 * Setup axios error interceptor for automatic error toast notifications.
 * Call this once at app startup (e.g., in main.tsx or App.tsx).
 */
export function setupErrorInterceptor() {
  axios.interceptors.response.use(
    // Pass through successful responses
    (response) => response,

    // Handle errors
    (error: AxiosError) => {
      const data = error.response?.data as any

      // Only show toast for backend error responses with status: 'error'
      if (data?.status === 'error') {
        const { addToast } = useToastStore.getState()

        // Determine if this is a GET request (retriable)
        const isGetRequest = error.config?.method?.toLowerCase() === 'get'

        addToast({
          title: data.message || 'An error occurred',
          message: data.error_id
            ? `Error ID: ${data.error_id}`
            : (data.code || undefined),
          color: 'red',
          icon: '✗',
          // Add retry action for GET requests only
          retry: isGetRequest && error.config
            ? () => axios(error.config!)
            : undefined
        })
      }

      // Always reject so callers can still handle errors locally
      return Promise.reject(error)
    }
  )
}
