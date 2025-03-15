import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 8000,
    watch: { usePolling: true },
    proxy: {
      '/api/process': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/api/analytics': {  // Add this proxy rule
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/api/content-analysis': { 
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/ws': {
        target: 'wss://api.ephremst.com',
        ws: true,
        changeOrigin: true,
        secure: true,
        headers: {
          'Origin': 'https://engageai.ephremst.com'
        }
      }
    },
    allowedHosts: [
      'engageai.ephremst.com',
      'api.ephremst.com',
      'localhost',
      '127.0.0.1'
    ]
  },
  preview: { host: '0.0.0.0', port: 8000 },
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx', '.jsx': 'jsx' }
    }
  }
})