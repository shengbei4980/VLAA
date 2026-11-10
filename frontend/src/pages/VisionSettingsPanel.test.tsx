// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VisionSettingsPanel } from "./VisionSettingsPanel";
import {
  visionApi,
  type VisionEndpoint,
  type VisionSettings,
} from "../api/vision";
vi.mock("../api/vision", () => ({
  visionApi: { settings: vi.fn(), saveSettings: vi.fn(), probe: vi.fn() },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
beforeEach(() => vi.resetAllMocks());
describe("vision settings fallback", () => {
  it("keeps the full basic form visible after GET 404 and preserves input after save failure", async () => {
    vi.mocked(visionApi.settings).mockRejectedValue(new Error("Not Found"));
    vi.mocked(visionApi.saveSettings).mockRejectedValue(
      new Error("Request failed (404)"),
    );
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<VisionSettingsPanel />));
    expect(container.textContent).toContain("请重启新版后端");
    expect(container.querySelectorAll("fieldset")).toHaveLength(2);
    expect(container.textContent).toContain("视觉模型接入");
    expect(container.textContent).toContain("候选提出者 · 自动找对象");
    expect(container.textContent).toContain("独立视觉检查者 · 检查标注");
    expect(container.textContent).toContain("测试图像与坐标");
    expect(container.textContent).not.toContain("图像合同");
    expect(container.textContent).not.toContain("能力探针");
    expect(container.textContent).toContain("远端 API");
    expect(container.textContent).toContain("本地模型 HTTP 服务");
    expect(container.textContent).toContain("2301 或以上即可按原尺寸发送");
    const endpoint =
      container.querySelector<HTMLInputElement>('input[type="url"]')!;
    const key = container.querySelector<HTMLInputElement>(
      'input[type="password"]',
    )!;
    const setInput = (input: HTMLInputElement, value: string) =>
      act(() => {
        Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          "value",
        )!.set!.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
      });
    setInput(endpoint, "http://127.0.0.1:9000/v1");
    setInput(key, "temporary-input");
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "保存视觉设置")!
        .click(),
    );
    expect(endpoint.value).toBe("http://127.0.0.1:9000/v1");
    expect(key.value).toBe("temporary-input");
    expect(container.textContent).not.toContain("temporary-input");
    expect(container.textContent).toContain("输入会保留");
    expect(visionApi.probe).not.toHaveBeenCalled();
    expect(visionApi.saveSettings).not.toHaveBeenCalled();
    act(() => root.unmount());
    container.remove();
  });
  it("merges only edited fields after recovery and preserves the server reviewer", async () => {
    const endpoint: VisionEndpoint = {
      enabled: true,
      mode: "remote",
      api_base: "https://server-proposer.example/v1",
      model: "saved-proposer",
      auth_scheme: "bearer",
      auth_header_name: "",
      coordinate_format: "normalized_1000",
      max_image_side: 2048,
      max_output_tokens: 8192,
      timeout_seconds: 120,
      multi_image: true,
      structured_output: "json_schema",
      api_key_configured: true,
    };
    const server: VisionSettings = {
      proposer: endpoint,
      reviewer: {
        ...endpoint,
        api_base: "https://server-reviewer.example/v1",
        model: "saved-reviewer",
        auth_scheme: "header",
        auth_header_name: "X-Api-Key",
      },
    };
    vi.mocked(visionApi.settings)
      .mockRejectedValueOnce(new Error("Not Found"))
      .mockResolvedValue(server);
    vi.mocked(visionApi.saveSettings).mockImplementation(async (value) => ({
      proposer: value.proposer,
      reviewer: value.reviewer,
    }));
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<VisionSettingsPanel />));
    const proposerUrl =
      container.querySelector<HTMLInputElement>('input[type="url"]')!;
    act(() => {
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(proposerUrl, "https://new-proposer.example/v1");
      proposerUrl.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "保存视觉设置")!
        .click(),
    );
    expect(visionApi.settings).toHaveBeenCalledTimes(2);
    const submitted = vi.mocked(visionApi.saveSettings).mock.calls[0][0];
    expect(submitted.proposer.api_base).toBe("https://new-proposer.example/v1");
    expect(submitted.proposer.model).toBe("saved-proposer");
    expect(submitted.proposer.max_image_side).toBe(2048);
    expect(submitted.reviewer).toMatchObject(server.reviewer);
    expect(submitted.reviewer.api_key).toBeUndefined();
    expect(
      container.querySelectorAll<HTMLInputElement>('input[type="url"]')[1]
        .value,
    ).toBe(server.reviewer.api_base);
    act(() => root.unmount());
    container.remove();
  });
});
