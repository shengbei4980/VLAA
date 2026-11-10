// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { AnnotationView } from "./App";
import type { ClassLabel, Dataset, ImageItem, SamPrediction } from "./types";

vi.mock("./annotation/StandaloneMaskEditor", () => ({
  StandaloneMaskEditor: ({
    onSamBox,
  }: {
    onSamBox: (box: unknown) => void;
  }) => (
    <button onClick={() => onSamBox({ x: 0, y: 0, width: 5, height: 5 })}>
      请求 SAM
    </button>
  ),
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("SAM response identity", () => {
  it("does not attach a response to the image opened while inference was pending", async () => {
    const dataset: Dataset = {
      id: "dataset",
      name: "images",
      task_type: "segment",
      description: null,
      image_count: 2,
      annotated_image_count: 0,
      class_count: 1,
      created_at: "",
      updated_at: "",
    };
    const image: ImageItem = {
      id: "image-1",
      dataset_id: "dataset",
      file_name: "1.png",
      width: 10,
      height: 10,
      split: "train",
      status: "unannotated",
      file_url: "/image",
      created_at: "",
      updated_at: "",
    };
    const classes: ClassLabel[] = [
      {
        id: "class-1",
        dataset_id: "dataset",
        class_index: 0,
        name: "object",
        color: "#22c55e",
        created_at: "",
        updated_at: "",
      },
    ];
    let resolve: (value: SamPrediction) => void = () => undefined;
    const onSam = vi.fn(
      () =>
        new Promise<SamPrediction>((callback) => {
          resolve = callback;
        }),
    );
    const onSave = vi.fn();
    const annotations: [] = [];
    const props = {
      dataset,
      classes,
      annotations,
      activeClassId: "class-1",
      onClassChange: vi.fn(),
      onBack: vi.fn(),
      onSave,
      onSam,
      onSamPoints: vi.fn(),
      busy: false,
    };
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    act(() => root.render(<AnnotationView {...props} image={image} />));
    act(() =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "请求 SAM")!
        .click(),
    );
    act(() =>
      root.render(
        <AnnotationView
          {...props}
          image={{ ...image, id: "image-2", file_name: "2.png" }}
        />,
      ),
    );
    await act(async () =>
      resolve({
        image_id: "image-1",
        class_id: "class-1",
        mask_id: "mask",
        polygon: [
          [0, 0],
          [5, 0],
          [5, 5],
        ],
        score: 1,
        backend_used: "ultralytics_sam",
        device: "cpu",
      }),
    );
    act(() =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "保存标注")!
        .click(),
    );
    expect(onSave).toHaveBeenCalledWith([]);
    act(() => root.unmount());
    container.remove();
  });
});
