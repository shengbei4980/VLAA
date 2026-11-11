// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { AnnotationView } from "./App";
import type { Annotation, ClassLabel, Dataset, ImageItem } from "./types";

const { canvasProps } = vi.hoisted(() => ({ canvasProps: vi.fn() }));
vi.mock("./api/vision", () => ({
  visionApi: { mask: vi.fn().mockResolvedValue({ rle: { size: [3, 3], counts: [0, 1, 8] } }) },
}));
vi.mock("./components/AnnotationReviewPanel", () => ({ AnnotationReviewPanel: () => null }));
vi.mock("./annotation/MaskReviewCanvas", () => ({
  MaskReviewCanvas: (props: { selectedId: string; onSelect: (id: string) => void }) => {
    canvasProps(props);
    return <p data-testid="selected-object">{props.selectedId}</p>;
  },
}));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it("selects a mask-only object's stable id from its sidebar row and maps canvas selection back", async () => {
  const dataset: Dataset = { id: "dataset", name: "正式标注", task_type: "segment", description: null, image_count: 1, annotated_image_count: 1, class_count: 1, created_at: "", updated_at: "" };
  const image: ImageItem = { id: "image", dataset_id: dataset.id, file_name: "image.png", width: 3, height: 3, split: "train", status: "annotated", file_url: "/image", annotation_revision: 3, created_at: "", updated_at: "" };
  const classes: ClassLabel[] = [{ id: "class", dataset_id: dataset.id, class_index: 0, name: "people", color: "#22c55e", created_at: "", updated_at: "" }];
  const annotations: Annotation[] = [1, 2, 3].map((index) => ({ id: `annotation-${index}`, object_id: `object-${index}`, dataset_id: dataset.id, image_id: image.id, class_id: "class", class_index: 0, label: "people", color: "#22c55e", type: "polygon", polygon: null, bbox: null, obb: null, mask_ref: `masks/${index}.json`, source: "manual", created_at: "", updated_at: "" }));
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => root.render(<AnnotationView dataset={dataset} image={image} classes={classes} annotations={annotations} activeClassId="class" onClassChange={vi.fn()} onBack={vi.fn()} onSave={vi.fn()} onSam={vi.fn()} onSamPoints={vi.fn()} busy={false} />));
    const rows = container.querySelectorAll<HTMLDivElement>(".shape-row");
    // Clicking the category control changes categories without selecting its row.
    act(() => rows[2].querySelector("select")!.click());
    expect(canvasProps.mock.lastCall![0].selectedId).toBe("");
    act(() => rows[2].querySelector<HTMLElement>(".shape-index")!.click());
    expect(rows[2].classList.contains("active")).toBe(true);
    expect(canvasProps.mock.lastCall![0].selectedId).toBe("object-3");
    act(() => (canvasProps.mock.lastCall![0] as { onSelect: (id: string) => void }).onSelect("object-1"));
    expect(rows[0].classList.contains("active")).toBe(true);
    expect(canvasProps.mock.lastCall![0].selectedId).toBe("object-1");
  } finally {
    act(() => root.unmount());
    container.remove();
  }
});
