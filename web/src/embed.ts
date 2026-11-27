/**
 * SigLIP image embeddings in the browser (transformers.js). Must be the same ONNX file the
 * catalog was built with: catalog/build_index.py --models onnx:<MODEL_ID>:<variant>.
 */

import { AutoProcessor, RawImage, SiglipVisionModel, env } from "@huggingface/transformers";

declare const __ORT_VERSION__: string;
// The runtime's .wasm is too big for Cloudflare static assets: load it from jsDelivr (see vite.config.ts).
if (env.backends.onnx.wasm) env.backends.onnx.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${__ORT_VERSION__}/dist/`;

export const MODEL_ID = "Xenova/siglip-base-patch16-224";
/** "q8" = vision_model_quantized.onnx (~95 MB), "fp16" = vision_model_fp16.onnx (~186 MB). */
export const MODEL_DTYPE = "q8" as const;

type Loaded = { processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>; model: SiglipVisionModel };
let loading: Promise<Loaded> | null = null;

/** Load once per page; `onProgress` gets 0-1 while the model downloads (instant when cached). */
export function loadEmbedder(onProgress?: (fraction: number) => void): Promise<Loaded> {
  loading ??= (async () => {
    const device = "gpu" in navigator ? "webgpu" : "wasm";
    const progress_callback = (p: { status: string; progress?: number }) => {
      if (p.status === "progress_total" && p.progress !== undefined) onProgress?.(p.progress / 100);
    };
    const [processor, model] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID),
      SiglipVisionModel.from_pretrained(MODEL_ID, { dtype: MODEL_DTYPE, device, progress_callback }).catch(() =>
        // WebGPU can be listed but unusable (old drivers): fall back to WebAssembly.
        SiglipVisionModel.from_pretrained(MODEL_ID, { dtype: MODEL_DTYPE, device: "wasm", progress_callback }),
      ),
    ]);
    onProgress?.(1);
    return { processor, model: model as SiglipVisionModel };
  })();
  loading.catch(() => (loading = null)); // allow a retry after a failed download
  return loading;
}

/** L2-normalised embedding per canvas (one crop each). */
export async function embedCanvases(canvases: OffscreenCanvas[] | HTMLCanvasElement[]): Promise<Float32Array[]> {
  const { processor, model } = await loadEmbedder();
  const out: Float32Array[] = [];
  for (const canvas of canvases) {
    const inputs = await processor(RawImage.fromCanvas(canvas));
    const { pooler_output } = (await model(inputs)) as { pooler_output: { data: Float32Array } };
    const v = Float32Array.from(pooler_output.data);
    const norm = Math.hypot(...v) || 1;
    out.push(v.map((x) => x / norm));
  }
  return out;
}
