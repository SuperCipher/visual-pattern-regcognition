import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { defineConfig, type Plugin } from "vite";

// onnxruntime-web's WebAssembly file (~27 MB) is over Cloudflare's 25 MiB per-file limit, so it is
// loaded from jsDelivr at the exact version transformers.js depends on (see src/embed.ts).
const require = createRequire(import.meta.url);
const ortEntry = createRequire(require.resolve("@huggingface/transformers")).resolve("onnxruntime-web");
let dir = dirname(ortEntry);
while (!dir.endsWith("onnxruntime-web")) dir = dirname(dir);
const ortVersion: string = JSON.parse(readFileSync(join(dir, "package.json"), "utf8")).version;

const dropWasm: Plugin = {
  name: "drop-ort-wasm",
  generateBundle(_, bundle) {
    for (const name of Object.keys(bundle)) if (name.endsWith(".wasm")) delete bundle[name];
  },
};

// Relative asset paths, so the build works from any sub-path (previews).
export default defineConfig({
  base: "./",
  define: { __ORT_VERSION__: JSON.stringify(ortVersion) },
  plugins: [dropWasm],
});
