import { useEffect, useState } from "react";
import { storageApi, type StorageSettings } from "../api/storage";

export function StorageStatusBanner({
  refreshToken,
  onOpen,
}: {
  refreshToken: number;
  onOpen: () => void;
}) {
  const [settings, setSettings] = useState<StorageSettings>();
  useEffect(() => {
    let active = true;
    const refresh = () => {
      Promise.resolve()
        .then(() => storageApi.settings())
        .then((result) => {
          if (active) setSettings(result);
        })
        .catch(() => undefined);
    };
    refresh();
    const timer = window.setInterval(() => {
      if (!document.hidden) refresh();
    }, 10000);
    window.addEventListener("focus", refresh);
    return () => {
      active = false;
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
    };
  }, [refreshToken]);
  if (!settings?.restart_required && !settings?.write_locked) return null;
  return (
    <div className="validation" role="status">
      <span>
        {settings.write_locked
          ? "存储位置正在切换，当前写入已锁定。"
          : "存储配置等待重启。"}
        待采用目录 {settings.pending?.data_dir ?? settings.current.data_dir}
      </span>
      <button className="button" onClick={onOpen}>
        查看存储状态与重启要求
      </button>
    </div>
  );
}
