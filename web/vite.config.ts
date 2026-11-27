import { defineConfig } from "vite";

// Relative asset paths, so the build works from any sub-path (Pages previews, artifact previews).
export default defineConfig({ base: "./" });
