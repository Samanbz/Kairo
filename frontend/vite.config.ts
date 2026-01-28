import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Needed for Docker
    port: 5173,
    watch: {
      usePolling: true // Needed for Docker primarily on non-Linux, but good for stability in containers
    }
  }
})
