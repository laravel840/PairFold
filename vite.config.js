import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const rootDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  // Relative asset URLs so the built app works from FastAPI (/) and file-adjacent servers.
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(rootDir, "index.html"),
        viewer: resolve(rootDir, "viewer.html"),
      },
    },
  },
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    open: "http://127.0.0.1:5173/",
    watch: {
      // Windows locks large/binary files; watching them crashes Vite (EBUSY)
      ignored: [
        "**/paper/**",
        "**/Paper/**",
        "**/pairfold/data/**",
        "**/pairfold/checkpoints/**",
        "**/pairfold/calibration/**",
        "**/*.pdf",
        "**/*.jsonl",
        "**/benchmark_pdbs/**",
        "**/benchmarks/pdbs/**",
        "**/benchmarks/results/**",
        "**/node_modules/**",
        "**/dist/**",
      ],
    },
    proxy: {
      // Same paths as production (UI + API share one origin when served by pairfold.server)
      "/predict": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
