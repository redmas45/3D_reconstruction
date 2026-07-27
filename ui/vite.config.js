import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied rather than called cross-origin so the browser treats video,
// image and event-stream responses as same-origin. Streaming must not be buffered.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            proxyRes.headers["cache-control"] = "no-cache";
          });
        },
      },
    },
  },
});
