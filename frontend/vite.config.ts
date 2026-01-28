import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const allowedHosts = [
    'dev1.flightcore.pl',
    'dev2.flightcore.pl',
    'dev3.flightcore.pl',
    'dev4.flightcore.pl',
    'dev5.flightcore.pl',
    'dev6.flightcore.pl',
    ...(env.VITE_ALLOWED_HOSTS?.split(',').map(h => h.trim()).filter(Boolean) || [])
  ]

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 5173,
      allowedHosts,
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/health': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
