import { request as apiRequest } from "./client";
import type { Annotation } from "../types";

export interface ImageReviewDetail {
  image_id: string;
  revision: number;
  fingerprint: string;
  review_status: string;
  annotations: Annotation[];
  draft?: { annotations?: Annotation[] } | null;
  receipt?: Record<string, unknown> | null;
  issues: Array<{ message: string; blocking?: boolean }>;
}
export interface DatasetReviewStatus {
  confirmed: number;
  pending: number;
  confirmed_empty: number;
  total: number;
  items: Array<{
    image_id: string;
    revision: number;
    review_status: string;
    annotation_count: number;
    contains_auto: boolean;
  }>;
}
async function request<T>(path: string, body?: unknown): Promise<T> {
  return apiRequest<T>(
    `/api/datasets${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
}
export const reviewApi = {
  status: (datasetId: string) =>
    request<DatasetReviewStatus>(`/${datasetId}/review-status`),
  image: (datasetId: string, imageId: string) =>
    request<ImageReviewDetail>(`/${datasetId}/images/${imageId}/review`),
  confirm: (
    datasetId: string,
    imageId: string,
    revision: number,
    fingerprint: string,
    confirmedEmpty: boolean,
    reason: string,
  ) =>
    request<ImageReviewDetail>(
      `/${datasetId}/images/${imageId}/review/confirm`,
      {
        expected_revision: revision,
        expected_fingerprint: fingerprint,
        confirmed_empty: confirmedEmpty,
        reason,
        actor: "local-user",
      },
    ),
};
