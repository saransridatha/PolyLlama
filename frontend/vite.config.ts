import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/admin/",
  plugins: [react()],
  build: {
    outDir: "../app/static/dashboard",
    emptyOutDir: true,
    sourcemap: false,
    assetsDir: "assets",
    target: "es2020",
  },
  server: {
    port: 5173,
    proxy: {
      "/admin/api": "http://127.0.0.1:8001",
    },
  },
});
