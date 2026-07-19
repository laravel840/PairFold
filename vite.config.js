import { defineConfig } from "vite";

export default defineConfig({
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
      ],
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
