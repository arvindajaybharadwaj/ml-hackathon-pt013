import react from '@vitejs/plugin-react'
import process from 'node:process'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_BACKEND_TARGET || 'http://127.0.0.1:8000'
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      proxy: {
        '/api': { target, changeOrigin: true, timeout: 300000, proxyTimeout: 300000 },
        '/health': { target, changeOrigin: true },
      },
    },
    preview: { host: '0.0.0.0', proxy: { '/api': target, '/health': target } },
  }
})
