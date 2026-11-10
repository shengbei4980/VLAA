import { request as apiRequest } from "./client";
import type { BBox, LLMAuthScheme } from "../types";
import type { MaskRle } from "../types/workflows";

export interface VisionEndpoint {
  enabled: boolean;
  mode: "remote" | "local";
  api_base: string;
  model: string;
  api_key_configured?: boolean;
  config_hash?: string;
  auth_scheme: LLMAuthScheme;
  auth_header_name: string;
  coordinate_format: "pixel" | "normalized_1" | "normalized_1000";
  max_image_side: number;
  max_output_tokens: number;
  timeout_seconds: number;
  multi_image: boolean;
  structured_output: "json_schema" | "json_object";
}

export type VisionSettings = Record<"proposer" | "reviewer", VisionEndpoint>;
export interface MaskArtifact {
  mask_ref: string;
  rle?: MaskRle;
  width: number;
  height: number;
  area: number;
}
export interface VisionProbe {
  ok: boolean;
  code: string;
  message: string;
  model: string;
  mode: string;
  remote_test_performed: boolean;
  coordinate_verified: boolean;
}
export interface SamMaskResult {
  image_id: string;
  class_id: string;
  mask_ref: string;
  polygon?: [number, number][] | null;
  score?: number | null;
  backend_used: string;
  device?: string | null;
}

async function request<T>(
  path: string,
  body?: unknown,
  method = "POST",
): Promise<T> {
  return apiRequest<T>(
    `/api/vision${path}`,
    body === undefined
      ? undefined
      : {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
}

export const visionApi = {
  settings: () => request<VisionSettings>("/settings"),
  saveSettings: (
    body: Record<
      "proposer" | "reviewer",
      VisionEndpoint & { api_key?: string }
    >,
  ) => {
    const editable = (endpoint: VisionEndpoint & { api_key?: string }) => {
      const {
        api_key_configured: _configured,
        config_hash: _hash,
        ...fields
      } = endpoint;
      return fields;
    };
    return request<VisionSettings>(
      "/settings",
      { proposer: editable(body.proposer), reviewer: editable(body.reviewer) },
      "PUT",
    );
  },
  probe: (role: "proposer" | "reviewer") =>
    request<VisionProbe>(`/probe/${role}`, {}),
  saveMask: (rle: MaskRle) => request<MaskArtifact>("/masks", { rle }),
  mask: (reference: string) =>
    request<MaskArtifact>(
      `/masks/${encodeURIComponent(reference.replace(/^masks\//, "").replace(/\.json$/, ""))}`,
    ),
  sam: (payload: {
    image_id: string;
    class_id: string;
    prompt_type: "box" | "point";
    box?: BBox;
    points?: Array<{ x: number; y: number; label: 0 | 1 }>;
  }) => request<SamMaskResult>("/sam/predict", payload),
};
