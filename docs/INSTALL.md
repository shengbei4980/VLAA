# VLAA 安装与启动

本文对应 v0.2.0 源码安装包。准备好 Python、Node.js 后，安装脚本会安装本工具依赖；视觉服务、API 密钥和 SAM 权重由使用者另行配置。

## 1. 准备环境

| 项目 | 要求 | 检查方式 |
| --- | --- | --- |
| 操作系统 | 当前一键安装与启动入口面向 Windows | 使用 PowerShell 或 CMD |
| Python | 3.12 | `py -3.12 --version` |
| Node.js | 20 或更新，推荐 22 LTS | `node --version` |
| npm | 随 Node.js 安装 | `npm.cmd --version` |
| 网络 | 首次安装依赖、首次下载所需模型时需要 | 确保能访问所用软件源及服务 |

可从 [Python 官网](https://www.python.org/downloads/)与 [Node.js 官网](https://nodejs.org/en/download)安装。安装 Node.js 后重新打开终端，使 PATH 生效。下载模型时应遵守模型提供方的许可条件。

CPU 可以运行本地任务。GPU 训练或分割需要与设备匹配的 PyTorch/CUDA 环境；安装脚本不会选择显卡驱动或专用 CUDA 版本。先用少量图片检查本机速度和内存占用。

## 2. 取得源码

打开 [GitHub 仓库](https://github.com/shengbei4980/VLAA)，点击“Code → Download ZIP”下载并解压，或使用 Git：

```powershell
git clone https://github.com/shengbei4980/VLAA.git
cd VLAA
```

请解压到完整的文件夹中再运行，确认根目录能看到 `install.cmd`、`start.cmd`、`run.py`、`backend` 和 `frontend`。

源码包不附带 Python/Node.js 安装程序、模型权重、数据集、API 密钥或个人运行记录。

## 3. 安装依赖

双击根目录的 **`install.cmd`**。首次安装可能需要较长时间，完成前保留窗口。安装器会：

1. 检查 Python 3.12、Node.js 与 npm。
2. 创建或复用本仓库的 `.venv`。
3. 在 `.venv` 中安装 `backend/requirements.txt`。
4. 在 `frontend` 中执行 `npm ci`。

也可以在源码目录运行：

```powershell
py -3.12 scripts/setup.py
```

只想检查环境时运行：

```powershell
py -3.12 scripts/setup.py --check
```

`--check` 不安装依赖、不创建目录；尚未安装时会提示缺少的组件。安装失败后可以排除网络或环境问题，再运行相同命令。

## 4. 启动与关闭

双击 **`start.cmd`**，或在根目录运行：

```powershell
py -3.12 run.py
```

启动后访问：

- [工作台](http://127.0.0.1:5173/)
- [后端 API 文档](http://127.0.0.1:8000/docs)

启动器会使用仓库内 `.venv` 运行后端、使用本地 Node.js 运行前端，并打开浏览器。保留启动窗口。在网页中先完成或取消当前训练、标注等任务，再按 **Ctrl+C** 关闭本次启动的服务。

识别到同一项目的健康服务时，启动器会复用它们。端口被其他服务占用时，启动器会报告问题；请确认占用程序的身份后处理。

## 5. 首次配置模型

1. 打开“设置 → 视觉模型接入”。分别启用**候选提出者**和**独立视觉检查者**。
2. 为每个角色选择“远端 API”或“本地模型 HTTP 服务”，填写模型名称、API Base URL 和必要的 API Key。模型必须支持图片输入；本地服务需预先启动。
3. 选择与服务一致的认证方式与模型原生坐标格式，保存，然后点击“测试图像与坐标”。已有密钥留空会保留。
4. 分割任务还需打开“SAM 辅助标注”，启用 SAM，填写已有本地权重文件的路径并选择设备和推理尺寸。工作流启动前需能找到该文件。仅供复核的框形回退建议不具备真实分割能力。
5. 回到工作流，用少量测试图检查类别和轮廓质量，再处理正式批次。

“任务助手（Agent LLM）”控制对话助手；“视觉模型接入”控制工作流的看图标注与检查，需按用途分别配置。接口兼容性、坐标测试和实际标注质量都需要使用你选择的服务验证。

## 6. 保存位置与升级

工作台默认使用本机 `data` 目录保存数据库、设置、日志和受管任务产物。存储设置可管理工作台数据位置；运行前的“项目与保存位置”决定该项目的 JSON、预览图、模型和评估文件输出位置。两者用途不同，操作见[使用教程](USER_GUIDE.md#项目与保存位置)。

升级前关闭任务与启动器，备份工作台数据目录、根目录的 `.ywa-storage.json`（如有）及项目结果目录。用新版源码后重新运行 `install.cmd`。数据目录中的配置可能包含 API 密钥，备份应保留在自己的受控位置。

## 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 找不到 Python 3.12 | 安装对应版本，重新打开终端，执行 `py -3.12 --version` |
| 找不到 Node.js / npm | 安装 Node.js，并重新打开终端；检查 `node --version` 和 `npm.cmd --version` |
| 提示缺少 `.venv` 或 Vite | 在同一源码目录重新运行 `install.cmd` |
| 下载依赖失败 | 检查网络、代理与软件源可用性，保留错误文本后重试安装 |
| `Port 5173 is already in use` | 先访问工作台地址；若是自己的旧 VLAA，回原启动窗口关闭后重新启动；其他程序占用时先确认用途 |
| 视觉设置或工作流接口出现 `Not Found` | 确认启动的是当前源码的后端；先停止旧启动窗口，再从当前目录运行 `start.cmd` |
| SAM 权重不存在或配置变化 | 检查权重文件路径、文件是否完整及设备；运行前重新预览，运行中不要替换本批已选权重 |
| GPU 不可用 | 检查驱动与 PyTorch/CUDA 匹配情况，或在设置/训练页选择 CPU 先验证流程 |
| 网页打不开或服务退出 | 查看工作台数据目录下 `logs/launcher-backend.log` 和 `logs/launcher-frontend.log` |

本项目默认仅绑定本机地址，按本地单用户场景设计。公开部署与多人访问需要额外的部署与权限设计。

[返回 README](../README.md) · [开始使用](USER_GUIDE.md)
