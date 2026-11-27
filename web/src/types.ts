/** What the pipeline hands the UI. Phase B fills it from a mock file; phase C from the real run. */

export type Box = readonly [number, number, number, number]; // 0-1000 normalised x1, y1, x2, y2

export interface SurfaceOut {
  box: Box;
  family: string;
  surfaceType: string;
}

export interface DecorMatch {
  code: string;
  name: string;
  category: string;
  url: string;
  swatch: string;
  score: number;
}

export interface GroupOut {
  family: string;
  /** Indexes into PipelineResult.surfaces. */
  members: number[];
  /** Best first; always at least one (never "no match"). */
  matches: DecorMatch[];
}

export interface PipelineResult {
  photo: string;
  surfaces: SurfaceOut[];
  groups: GroupOut[];
}

export type Step = "model" | "surfaces" | "matching";

export interface Progress {
  step: Step;
  /** 0-1 when known (model download), else undefined. */
  fraction?: number;
}
