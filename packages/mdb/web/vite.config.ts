/// <reference types="vitest" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Built assets ship inside the Python wheel (mdb/serve/static); `mdb serve` serves them.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../src/mdb/serve/static',
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0,
  },
  // `mdb serve` refuses a Host that is not its own loopback address and any Origin that is not
  // its own. The dev proxy therefore rewrites Host to the target (changeOrigin) and drops the
  // browser's Origin (which names the Vite port). Start the server with `--port 8765`, or change
  // the target below. The built bundle served by `mdb serve` needs none of this.
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        ws: true,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (req) => req.removeHeader('origin'));
          proxy.on('proxyReqWs', (req) => req.removeHeader('origin'));
        },
      },
    },
  },
  test: { environment: 'node', include: ['src/**/*.test.{ts,tsx}'] },
});
