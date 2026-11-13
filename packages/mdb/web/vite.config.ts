/// <reference types="vitest" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { DEV_PORT, TARGET, guardDevProxy } from './src/dev/proxyGuard';

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
  // its own. The dev proxy rewrites Host to the target (changeOrigin) and rewrites the browser's
  // Origin only when it is this dev server's own (src/dev/proxyGuard.ts); any other Origin is
  // refused 403 here. Start the server with `--port 8765`, or change TARGET there. The built
  // bundle served by `mdb serve` needs none of this.
  server: {
    port: DEV_PORT,
    strictPort: true,
    proxy: {
      '/api': {
        target: TARGET,
        ws: true,
        changeOrigin: true,
        configure: (proxy) =>
          guardDevProxy(proxy as unknown as Parameters<typeof guardDevProxy>[0]),
      },
    },
  },
  test: { environment: 'node', include: ['src/**/*.test.{ts,tsx}'] },
});
