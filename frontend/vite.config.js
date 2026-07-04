import { defineConfig } from 'vite'

export default defineConfig({
  root: '.',
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api': 'http://localhost:8010',
      '/solve': 'http://localhost:8010',
      '/feedback': 'http://localhost:8010',
      '/sessions': 'http://localhost:8010',
      '/health': 'http://localhost:8010',
      '/pipeline': 'http://localhost:8010',
    }
  },
  build: {
    outDir: 'dist',
  }
})
