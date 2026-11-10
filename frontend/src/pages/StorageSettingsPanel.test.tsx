// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { StorageSettingsPanel } from "./StorageSettingsPanel";
import { storageApi, type StorageSettings } from "../api/storage";
vi.mock("../components/ExportLocationsPanel", () => ({
  ExportLocationsPanel: () => <div>JSON 与 PNG 保存位置</div>,
}));
vi.mock("../api/storage", () => ({
  storageApi: {
    settings: vi.fn(),
    validateDestination: vi.fn(),
    registerDestination: vi.fn(),
    prepareDataRoot: vi.fn(),
    executeDataRoot: vi.fn(),
    cancelDataRoot: vi.fn(),
  },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
const current: StorageSettings = {
  current: {
    data_dir: "D:\\old-data",
    database_path: "D:\\old-data\\app.db",
    source: "config",
  },
  pending: null,
  restart_required: false,
  write_locked: false,
  environment_overrides: { data_dir: false, database_url: false },
  destinations: [],
  migration: null,
};
const inputValue = (input: HTMLInputElement, value: string) =>
  act(() => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

describe("storage locations", () => {
  it("validates a registered destination before saving its resolved absolute path", async () => {
    vi.clearAllMocks();
    vi.mocked(storageApi.settings).mockResolvedValue(current);
    vi.mocked(storageApi.validateDestination).mockResolvedValue({
      resolved_path: "E:\\results",
      writable: true,
      free_bytes: 1000,
    });
    vi.mocked(storageApi.registerDestination).mockResolvedValue({
      id: "dest",
      name: "Results",
      path: "E:\\results",
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<StorageSettingsPanel />));
    const inputs = container.querySelectorAll<HTMLInputElement>("input");
    inputValue(inputs[0], "Results");
    inputValue(inputs[1], "E:\\results");
    const button = (name: string) =>
      Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
        (item) => item.textContent === name,
      )!;
    expect(button("注册输出目标").disabled).toBe(true);
    await act(async () => button("验证目录").click());
    await act(async () => button("注册输出目标").click());
    expect(storageApi.registerDestination).toHaveBeenCalledWith(
      "Results",
      "E:\\results",
    );
    expect(container.textContent).toContain("输出目标已注册");
    act(() => root.unmount());
    container.remove();
  });
  it("requires copy-plan approval, then exposes pending root and restart requirement", async () => {
    vi.clearAllMocks();
    vi.mocked(storageApi.settings).mockResolvedValue(current);
    vi.mocked(storageApi.prepareDataRoot).mockResolvedValue({
      plan_id: "plan",
      source_dir: "D:\\old-data",
      target_dir: "E:\\new-data",
      database_path: "E:\\new-data\\app.db",
      required_bytes: 100,
      free_bytes: 10000,
      file_count: 4,
      restart_required: true,
    });
    vi.mocked(storageApi.executeDataRoot).mockResolvedValue({
      ...current,
      pending: {
        data_dir: "E:\\new-data",
        database_path: "E:\\new-data\\app.db",
      },
      restart_required: true,
      write_locked: true,
    });
    vi.mocked(storageApi.cancelDataRoot).mockResolvedValue(current);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<StorageSettingsPanel />));
    const inputs = container.querySelectorAll<HTMLInputElement>("input");
    inputValue(inputs[2], "E:\\new-data");
    const button = (name: string) =>
      Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
        (item) => item.textContent === name,
      )!;
    await act(async () => button("预检数据根迁移").click());
    expect(button("执行复制迁移").disabled).toBe(true);
    expect(storageApi.executeDataRoot).not.toHaveBeenCalled();
    act(() =>
      container
        .querySelector<HTMLInputElement>('input[type="checkbox"]')!
        .click(),
    );
    await act(async () => button("执行复制迁移").click());
    expect(storageApi.executeDataRoot).toHaveBeenCalledWith("plan");
    expect(container.textContent).toContain("需要重启工作台使新目录生效");
    expect(container.textContent).toContain("当前写入已锁定");
    expect(container.textContent).toContain("D:\\old-data");
    await act(async () => button("撤销待切换配置").click());
    expect(storageApi.cancelDataRoot).toHaveBeenCalled();
    expect(container.textContent).toContain("复制目录保留");
    act(() => root.unmount());
    container.remove();
  });
});
