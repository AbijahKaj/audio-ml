import { defineConfig, type Plugin } from 'vite';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * Vite plugin that serves tfjs .wasm binaries from node_modules with the
 * correct MIME type. Without this, the Vite dev server returns HTML for
 * unknown paths, which causes WebAssembly.compile to fail.
 */
function tfjsWasm(): Plugin {
  const wasmDir = resolve(__dirname, '../node_modules/@tensorflow/tfjs-backend-wasm/dist');

  return {
    name: 'tfjs-wasm',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/tfjs-wasm/') && req.url.endsWith('.wasm')) {
          const filename = req.url.replace('/tfjs-wasm/', '');
          const filepath = resolve(wasmDir, filename);
          if (existsSync(filepath)) {
            res.setHeader('Content-Type', 'application/wasm');
            res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
            res.end(readFileSync(filepath));
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [tfjsWasm()],
  server: {
    fs: {
      allow: ['..'],
    },
  },
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-wasm',
    ],
  },
});
