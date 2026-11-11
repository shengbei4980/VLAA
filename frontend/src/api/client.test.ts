import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, request } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("HTTP errors", () => {
  it.each([
    [404, { detail: "Not Found" }, "Not Found"],
    [409, { error: { message: "版本已更新" } }, "版本已更新"],
    [422, { detail: [{ loc: ["body", "revision"], msg: "无效" }] }, "body.revision 无效"],
  ])("preserves HTTP status and existing error messages (%s)", async (status, body, message) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status, json: async () => body }));
    const error = await request("/test").catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status, message });
  });

  it("preserves the status when the backend returns a non-JSON error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => { throw new Error("Invalid JSON"); } }));
    await expect(request("/test")).rejects.toMatchObject({ status: 502, message: "Request failed (502)" });
  });
});
