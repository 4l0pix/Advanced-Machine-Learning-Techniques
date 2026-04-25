import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Sometimes react-syntax-highlighter needs help finding its ESM entry
      'react-syntax-highlighter': 'react-syntax-highlighter/dist/esm/index.js',
    },
  },
  optimizeDeps: {
    include: ['react-syntax-highlighter', 'framer-motion'],
  },
});
