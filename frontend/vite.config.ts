import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Target modern browsers for smaller bundles
    target: 'es2020',
    // Enable source maps only in dev
    sourcemap: false,
    // Split chunks for optimal caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Core React runtime — cached forever
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // UI framework
          'vendor-ui': ['@radix-ui/react-tooltip', '@radix-ui/react-select', '@radix-ui/react-tabs', '@radix-ui/react-separator', '@radix-ui/react-scroll-area', '@radix-ui/react-checkbox', '@radix-ui/react-label', '@radix-ui/react-avatar'],
          // Data layer
          'vendor-data': ['axios', '@tanstack/react-query', 'zustand'],
          // Charts (large dependency)
          'vendor-charts': ['recharts'],
          // Utilities
          'vendor-utils': ['date-fns', 'lucide-react', 'clsx', 'tailwind-merge'],
        },
      },
    },
    // Increase chunk size warning limit
    chunkSizeWarningLimit: 600,
  },
})
