# YoloWebAgentStarter 贡献说明

YoloWebAgentStarter 是 YoloWebAgent 的独立社区版。运行时不得 import、读取或依赖 Enterprise 仓库。

## 产品边界

- Community v2 支持 `detect`、`segment`、`obb` 与 `classify`；其中 polygon 是 segment 的标注表示。
- 核心流程为图片与任务配置档 → 候选标注 → 检查与修正 → 人工确认 → 冻结快照 → 导出/训练/评估及下一批。segment 使用原始 RLE mask，兼容已有 polygon。SAM 支持交互建议与受控工作流分割，候选由用户确认后正式应用。
- 训练在本地以队列方式运行；模型版本只能由受管训练产物创建。本地评估复用已持久化 split，对受管 PT 运行原生 YOLO `val`。
- Agent MVP 在范围内：只读智能助手（数据集/训练/模型问答与报告），以及经人工确认后调用现有领域 Service 提交受管任务（训练、评估、自动标注）。工具调用与运行记录可持久化；LLM 配置可在设置页写入 `{data_dir}/settings.json`（与上游一致），也可由环境变量提供默认值；不得写入 SQLite。
- 通用图像工作流已按用户确认的 2026-11-09 修订方案纳入范围，使用版本化模板、任务配置档、白名单节点、人工关卡及有界继续条件。视觉服务支持显式配置的远端 API 或本地端点，凭据保存在本地设置。训练、评估及发布导出统一使用具有当前人工回执的冻结数据。
- Auth、RBAC、License、Deployment、pose、定时触发与远程部署分别按后续需求处理。Agent 的执行边界为已注册的领域服务与受管产物。
- 服务仅面向本地单用户，默认绑定 `127.0.0.1`。

## 工程规则

- Python 依赖必须安装在仓库的 `.venv` 中；不得安装到系统 Python。
- API 路由只负责适配 HTTP 请求；业务逻辑必须位于领域服务中。
- 文件 IO 必须经由 `backend/app/core/storage.py` 或明确命名的领域边界。
- 标注坐标必须以图片绝对像素持久化；Canvas 状态不是持久化 schema。
- 校验与 YOLO 导出必须复用已持久化的图片 split。
- 新 schema 变更必须提供 Alembic migration；不得添加运行时 SQLite patcher。
- 前端 API 调用必须置于 `frontend/src/api/`；页面组件不得硬编码 API URL。
- 在 `source_snapshot.md` 中保留来源归属；不得复制 Enterprise 的 `.git` 目录或 migration 历史。

## 验证

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```
