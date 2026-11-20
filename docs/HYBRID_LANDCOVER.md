# Sentinel-2 遥感联合标注

在 VLAA 主导航选择 **遥感联合标注**。使用默认启动器时，可以直接打开本机的 [联合标注页面](http://127.0.0.1:5173/#hybrid)。每次运行处理一张 RGB 图块。输入和结果的坐标都是该图块的原始像素坐标。

## 准备输入

| 文件 | 要求 |
|---|---|
| RGB 图块 | PNG 或 JPEG，保留原始尺寸与方向。使用原图文件名，以便导入已有草稿或回到标注器校对。 |
| Sentinel-2 地表反射率 | 带坐标参考系的 GeoTIFF，波段说明可识别 B02、B03、B04、B05、B06、B07、B08、B8A、B11、B12。程序按波段名称取数。整数 DN 需有 `reflectance_scale_factor` 或 GDAL 波段比例元数据，未缩放的浮点数据应已是真实反射率。 |
| X、Y 偏移 | 输入整景 TIFF 时，X 是图块左上角相对**该 TIFF**的列号，Y 是行号，均从 0 开始。若图块先从整景切为母图、再从母图切出，两个层级的偏移要相加。输入与整张 RGB 已配准的同尺寸小 TIFF 时，两值均填 0。 |
| 有效范围（可选） | 与 RGB 同尺寸的单通道 PNG，非零像素参与处理。NoData 和非有限光谱值排除。 |

先确认影像日期、空间分辨率和像素网格一致。普通 PNG/JPEG 无法独立证明与地理栅格已配准。页面按填写的位置读取窗口，不会自动做地理配准。当前单张上限为 4,194,304 像素，整景 TIFF 上传上限为 512 MiB；对于更大的整景，可预先裁出同坐标、保留波段说明和定标元数据的 TIFF。

## 选择生成方式

- **导入两路已有草稿**。选择 GPT 直接多边形 LabelMe JSON 和同一张图的 DeepSeek＋SAM 掩膜 LabelMe JSON。两份草稿应保留原图名和尺寸，类别采用 `water`、`bare`、`grass`、`wood`、`crop`、`built`。这一方式在本机融合，只需仓库随附的随机森林。
- **从原图调用视觉模型**。先在 VLAA 设置里配置并启用可用的本地 SAM 权重，再在本页为两个角色分别指定服务。GPT 用于提出带类别的多边形，DeepSeek 用于提出带类别的候选框。各角色可选择已保存的视觉端点，也可为本次运行填写 Base URL、模型名和 API Key。选择服务实际支持的协议与坐标约定。支持非流式 Responses 和 Chat Completions。确认发送后，两个视觉端点各接收本张 RGB 图；十波段数据留在本地，候选框由本地 SAM 生成掩膜。

## 如何融合和校对

1. GPT 先标记有把握的地块。主标已给出类别的位置保留，已有的跨类重叠作为冲突留待人工处理。
2. 在 GPT 留白处，若 SAM 掩膜支持唯一类别，则补入该类。多个对象同属一个类别时，按一个类别计算。
3. 对仍然空白的有效像素，十波段随机森林按光谱给出类别。近期人工校对时，优先核对来源预览中的 RF 新增区、SAM 边界和冲突，尤其是草本、裸地与耕地。
4. 点 **生成待校对草稿**，在结果区切换「地表类别」与「标注来源」，下载同名 LabelMe JSON。要在 VLAA 中继续修边和确认，把原图和 JSON 配对、打包为 LabelMe ZIP，在数据集的导入中心预览并导入。JSON 中保留分层来源；下载结果仍是待审草稿。

随仓库提供的模型是以潘安湖 2025 年单张人工标注母图训练的完整随机森林。六图开发对照使用空间留出模型计分；将完整模型再用于相同六张图所得的结果不应作为独立测试。换地块或年份前可用当地独立人工样本检查每类表现。[模型信息与校验值](../backend/app/hybrid_landcover/weights/README.md)

## English quick start

Open **Hybrid land-cover annotation** from the main navigation. Supply an RGB tile and a georeferenced, band-labelled Sentinel-2 surface-reflectance GeoTIFF. With a full scene, enter the tile’s zero-based column and row offsets in that scene. For an aligned tile-sized TIFF, enter zero for both. An optional single-channel PNG limits the eligible area.

Choose **Import two existing drafts** to run locally with matched GPT polygon and DeepSeek/SAM mask JSON files. Alternatively, **Generate with vision endpoints** uses one polygon proposal and one box proposal from the configured services, then segments the boxes with local SAM. The bundled forest fills remaining eligible gaps. Inspect the class and source overlays, download the LabelMe JSON and import it with the original image for human review.
