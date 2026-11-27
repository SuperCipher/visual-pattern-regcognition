/** Phase B stand-in for the pipeline: replays a saved result with realistic pauses. */

import type { PipelineResult, Progress } from "./types";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function runMock(onProgress: (p: Progress) => void): Promise<PipelineResult> {
  for (let i = 0; i <= 10; i++) {
    onProgress({ step: "model", fraction: i / 10 });
    await sleep(80);
  }
  onProgress({ step: "surfaces" });
  const res = await fetch(`${import.meta.env.BASE_URL}mock/example.json`);
  if (!res.ok) throw new Error(`mock data missing (${res.status})`);
  const data = (await res.json()) as PipelineResult;
  await sleep(1400);
  onProgress({ step: "matching" });
  await sleep(700);
  return data;
}
