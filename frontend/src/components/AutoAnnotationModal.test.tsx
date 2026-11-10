// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { AutoAnnotationModal } from "./AutoAnnotationModal";
import { api } from "../api/client";
import type { AutoAnnotationTask, Dataset, ModelVersion } from "../types";
vi.mock("../api/client", () => ({
  api: {
    listModels: vi.fn(),
    listClasses: vi.fn(),
    createAutoAnnotation: vi.fn(),
  },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("explicit auto annotation class mapping", () => {
  it("sends an ignored same-name class as null and preserves formal annotations", async () => {
    const dataset: Dataset = {
      id: "target",
      name: "target",
      task_type: "detect",
      description: null,
      image_count: 1,
      annotated_image_count: 0,
      class_count: 2,
      created_at: "",
      updated_at: "",
    };
    vi.mocked(api.listModels).mockResolvedValue({
      total: 1,
      items: [
        {
          id: "model",
          name: "model",
          status: "active",
          format: "pt",
          task_type: "detect",
          dataset_id: "source",
          artifact_type: "best",
        } as ModelVersion,
      ],
    });
    vi.mocked(api.listClasses).mockImplementation(async (id) => [
      {
        id: `${id}-a`,
        dataset_id: id,
        name: "tower",
        class_index: 0,
        color: "#22c55e",
        created_at: "",
        updated_at: "",
      },
      {
        id: `${id}-b`,
        dataset_id: id,
        name: "window",
        class_index: 1,
        color: "#22c55e",
        created_at: "",
        updated_at: "",
      },
    ]);
    vi.mocked(api.createAutoAnnotation).mockResolvedValue({
      id: "task",
      status: "completed",
      processed_images: 1,
      total_images: 1,
      progress_percent: 100,
      created_annotations: 1,
      skipped_images: 0,
      confidence: 0.25,
    } as AutoAnnotationTask);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <AutoAnnotationModal
          dataset={dataset}
          onClose={vi.fn()}
          onChanged={vi.fn()}
        />,
      ),
    );
    const mapping = container.querySelector<HTMLSelectElement>(
      ".auto-annotation-mapping select",
    )!;
    act(() => {
      mapping.value = "";
      mapping.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "开始自动标注")!
        .click(),
    );
    expect(api.createAutoAnnotation).toHaveBeenCalledWith(
      "target",
      expect.objectContaining({
        class_mapping: { "0": null, "1": "target-b" },
        clean_old_annotations: false,
      }),
    );
    expect(container.textContent).toContain("待审区");
    act(() => root.unmount());
    container.remove();
  });
});
