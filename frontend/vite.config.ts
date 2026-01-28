import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from "path"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: true, // Needed for Docker
    port: 5173,
    watch: {
      usePolling: true // Needed for Docker primarily on non-Linux, but good for stability in containers
    }
  }
})
