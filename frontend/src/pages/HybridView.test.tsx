// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildHybridPayload, hybridApi, type HybridResult } from "../api/hybrid";
import { HybridView } from "./HybridView";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("hybrid land-cover workspace", () => {
  let root: Root;
  let host: HTMLDivElement;
  beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    vi.spyOn(hybridApi, "status").mockResolvedValue({ rf_ready: true, rf_detail: "ready", sam_ready: false, sam_detail: "not configured" });
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("allows imported masks without SAM and requires SAM for online generation", async () => {
    await act(async () => root.render(<HybridView locale="en" />));
    expect(host.querySelector("h1")?.textContent).toBe("Hybrid land-cover annotation");
    const generate = host.querySelector<HTMLButtonElement>("button[type=submit]")!;
    expect(generate.disabled).toBe(false);
    expect(host.querySelector("input[name=sam_draft]")).toBeTruthy();
    const online = host.querySelectorAll<HTMLInputElement>("input[name=mode]")[1];
    await act(async () => online.click());
    expect(generate.disabled).toBe(true);
    expect(host.querySelector("input[name=sam_draft]")).toBeNull();
    expect(host.querySelector<HTMLInputElement>("input[name=allow_remote]")?.required).toBe(true);
    expect(host.querySelectorAll("input[type=password]")).toHaveLength(2);
    const source = host.querySelector<HTMLSelectElement>("select[name=gpt_source]")!;
    await act(async () => {
      source.value = "proposer";
      source.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(host.querySelectorAll("input[type=password]")).toHaveLength(1);
  });

  it("opens class and source previews and downloads the LabelMe result", async () => {
    const rgb = new File(["rgb"], "tile_x0012_y0020.png", { type: "image/png" });
    const result: HybridResult = {
      labelme: { imagePath: rgb.name, shapes: [] }, preview_png: "class", source_png: "source",
      stats: { gpt_pixels: 2, sam_pixels: 3, rf_pixels: 4, eligible_pixels: 10, conflict_pixels: 1 }, warnings: [],
    };
    const nativeFormData = FormData;
    // jsdom cannot populate a native FileList via a user file chooser.
    vi.stubGlobal("FormData", class extends nativeFormData {
      constructor(form?: HTMLFormElement) {
        super(form);
        if (form) {
          this.set("rgb", rgb);
          for (const name of ["raster", "gpt_draft", "sam_draft"]) this.set(name, new File(["x"], name));
        }
      }
    });
    const objectUrl = vi.fn(() => "blob:local-preview");
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = objectUrl;
      static revokeObjectURL = vi.fn();
    });
    vi.spyOn(hybridApi, "run").mockResolvedValue(result);
    await act(async () => root.render(<HybridView locale="zh" />));
    const form = host.querySelector("form")!;
    vi.spyOn(form, "reportValidity").mockReturnValue(true);
    await act(async () => form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true })));
    const captured = vi.mocked(hybridApi.run).mock.calls[0][0];
    expect(captured.get("mode")).toBe("drafts");
    expect(captured.has("gpt_endpoint")).toBe(false);
    expect(captured.has("roi")).toBe(false);
    expect(host.querySelector("a[download]")?.getAttribute("download")).toBe("tile_x0012_y0020.json");
    const layer = host.querySelector<HTMLSelectElement>("section[aria-label='草稿结果'] select")!;
    await act(async () => {
      layer.value = "source";
      layer.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(host.querySelector("img[alt=来源预览]")?.getAttribute("src")).toBe("data:image/png;base64,source");
    expect(host.textContent).toContain("RF · 4 px");
  });

  it("sends saved credentials only by role and custom credentials only to the intended role", () => {
    const fields = new FormData();
    const values = { x: "128", y: "64", allow_remote: "on",
      gpt_source: "proposer", gpt_url: "https://old.invalid", gpt_key: "must-not-be-forwarded", gpt_protocol: "responses",
      deepseek_source: "custom", deepseek_url: "https://vision.invalid/v1", deepseek_model: "vision", deepseek_key: "fake-per-request",
      deepseek_coords: "normalized_1000", deepseek_protocol: "chat_completions" };
    for (const [field, value] of Object.entries(values)) fields.set(field, value);
    const payload = buildHybridPayload(fields, "online");
    expect(payload.get("allow_remote")).toBe("true");
    expect(JSON.parse(String(payload.get("gpt_endpoint")))).toEqual({ saved_role: "proposer", protocol: "responses" });
    expect(JSON.parse(String(payload.get("deepseek_endpoint")))).toEqual({
      model: "vision", api_base: "https://vision.invalid/v1", api_key: "fake-per-request",
      coordinate_format: "normalized_1000", protocol: "chat_completions",
    });
    expect(payload.has("gpt_key")).toBe(false);
  });
});
