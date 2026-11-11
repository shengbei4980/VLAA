import { afterEach, describe, expect, it, vi } from "vitest";
import { workflowApi } from "./workflows";
import { visionApi, type VisionEndpoint } from "./vision";

afterEach(() => vi.unstubAllGlobals());
describe("workflow HTTP contracts", () => {
  it("requests a manual revision of the confirmed review without rerunning a model", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "new-review", run_id: "manual-run" }) });
    vi.stubGlobal("fetch", fetch);
    const result = await workflowApi.continueEditing("review/one", 3);
    expect(result).toEqual({ id: "new-review", run_id: "manual-run" });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toContain("/api/workflows/reviews/review%2Fone/continue-editing");
    expect(fetch.mock.calls[0][1].method).toBe("POST");
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ expected_revision: 3 });
  });
  it("polls lightweight summaries instead of downloading annotation payloads", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    await workflowApi.run("run/one");
    expect(fetch.mock.calls[0][0]).toContain("/api/workflows/runs/run%2Fone?summary=true");
  });
  it("deletes and restores run records through explicit HTTP methods", async () => {
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    await workflowApi.runs(true);
    await workflowApi.deleteRun("run/one");
    await workflowApi.restoreRun("run/one");
    expect(fetch.mock.calls[0][0]).toContain("/api/workflows/runs?deleted=true");
    expect(fetch.mock.calls[1][0]).toContain("/api/workflows/runs/run%2Fone");
    expect(fetch.mock.calls[1][1].method).toBe("DELETE");
    expect(fetch.mock.calls[2][0]).toContain("/api/workflows/runs/run%2Fone/restore");
    expect(fetch.mock.calls[2][1].method).toBe("POST");
  });
  it("registers project roots and exports batch previews through encoded workflow endpoints", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    await workflowApi.projects();
    await workflowApi.saveProject({ name: "塔体", root_path: "E:\\塔体项目" });
    await workflowApi.exportBatch("run/one");
    expect(fetch.mock.calls[0][0]).toContain("/api/workflows/projects");
    expect(fetch.mock.calls[0][1]).toBeUndefined();
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({
      name: "塔体",
      root_path: "E:\\塔体项目",
    });
    expect(fetch.mock.calls[2][0]).toContain(
      "/api/workflows/runs/run%2Fone/annotation-batch",
    );
    expect(JSON.parse(fetch.mock.calls[2][1].body)).toEqual({
      include_previews: true,
    });
  });
  it("writes review drafts without submitting a decision", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: "review-1", revision: 2 }),
    });
    vi.stubGlobal("fetch", fetch);
    await workflowApi.saveDraft("review-1", 1, {
      image_id: "image-1",
      dataset_id: "dataset-1",
      image_url: "/image",
      width: 10,
      height: 10,
      base_revision: 0,
      objects: [],
      issues: [],
    });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toContain(
      "/api/workflows/reviews/review-1/draft",
    );
    expect(fetch.mock.calls[0][1].method).toBe("PUT");
    expect(JSON.parse(fetch.mock.calls[0][1].body).expected_revision).toBe(1);
  });
  it("keeps secret-status, fingerprints and legacy image limits out of settings writes", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetch);
    const endpoint: VisionEndpoint & { max_image_side: number } = {
      enabled: false,
      mode: "local",
      api_base: "http://127.0.0.1:8001/v1",
      model: "vision",
      auth_scheme: "bearer",
      auth_header_name: "",
      coordinate_format: "pixel",
      max_image_side: 1024,
      max_output_tokens: 4096,
      timeout_seconds: 60,
      multi_image: true,
      structured_output: "json_object",
      api_key_configured: true,
      config_hash: "safe-fingerprint",
    };
    await visionApi.saveSettings({ proposer: endpoint, reviewer: endpoint });
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.proposer.api_key_configured).toBeUndefined();
    expect(body.proposer.config_hash).toBeUndefined();
    expect(body.proposer.api_key).toBeUndefined();
    expect(body.proposer.max_image_side).toBeUndefined();
    expect(body.reviewer.max_image_side).toBeUndefined();
    expect(body.proposer.model).toBe("vision");
  });
});
