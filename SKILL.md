---
name: talking-head-music-edit
description: 口播精剪与视频配乐技能。当用户需要精剪口播视频并生成字幕、为口播文案选曲配乐、或单独剪辑一首背景音乐并交付剪映草稿 / ADu 工程 / FFmpeg 成片时使用。
---

# 口播精剪与配乐

## 分支路由

按用户当次目标选择唯一分支，只走该分支经过的阶段并只读取对应的参考文件：

| 分支 | 执行阶段 | 按需读取 |
| --- | --- | --- |
| **只剪口播** | 阶段 0 → 阶段 1 → 阶段 3 | `references/00_初始化与素材转写.md`、`references/speech/*`、`references/01_时间线执行格式与交付.md` |
| **纯配乐 / 单曲剪辑** | 阶段 0 → 阶段 2 → 阶段 3 | `references/00_初始化与素材转写.md`、`references/music/*`、`references/01_时间线执行格式与交付.md` |
| **口播 + 配乐** | 阶段 0 → 阶段 1 → 阶段 2 → 阶段 3 | 全部 `references/` |

---

## 阶段 0：初始化与配置核验

读取 [00_初始化与素材转写](references/00_初始化与素材转写.md) 完成环境检查、配置保存与音乐库整理。

1. 运行 `python -B scripts/init_machine.py --inspect`。
2. 针对 `missing_required_fields` 逐项向用户确认并写入本机配置（剪映真实草稿目录、逐字识别方式、音乐库目录、默认档位与音量偏好）。
3. 本次涉及配乐时，运行 `python -B scripts/check_music_library.py --dir "<音乐库>" --check`；若 `compliant` 为 `false`，征得同意后运行 `--organize` 归位建索引。

**完成标准**：`init_machine.py --inspect` 返回 `initialized_complete: true`；涉及配乐时音乐库检查返回 `compliant: true`。全部满足才进入下一阶段。

---

## 阶段 1：口播识别、方案确认与精剪

1. 按 [00_初始化与素材转写](references/00_初始化与素材转写.md) 在视频原文件夹产出 `<视频名>.srt`、`<视频名>_逐字时间.json` 与 `scan_report.json`（含自动计算的 `speech_rate` 语速评分与加速建议）。
2. 读取 [01_剪辑档位与语速标准](references/speech/01_剪辑档位与语速标准.md) 与 [02_去重纠错与刀口气口](references/speech/02_去重纠错与刀口气口.md)，合并连续流畅句为完整长片段，拟定删减与纠错方案。
3. **按简洁模板在对话正文汇报并请用户确认**：必须包含「语速评分与加速建议」「3 种口播档位选择（1 平衡型 / 2 专业紧凑型 / 3 人性化口播型）」「错字纠错清单」「剪辑概览（原长→预计成片、保留几大段/切几刀、主要删减）」。命中已保存类型习惯且 `auto_confirm: true` 时，默认标出所选档位并简要展示摘要。
4. 用户确认档位、倍速与方案后，写出 `timeline_edit.json`，将新纠错词写入本机常用词汇表，并按 [03_字幕规则与跨切点](references/speech/03_字幕规则与跨切点.md) 运行 `scripts/subtitle_processor.py` 产出精剪 `.srt` 与纯净正文 `.txt`。

**完成标准**：用户已确认口播档位、倍速与剪辑方案；视频原文件夹已落盘 `timeline_edit.json`、精剪 `.srt` 与纯净 `.txt`；`python -B scripts/execute_timeline_edit.py --plan "<timeline_edit.json>" --check` 通过且无切碎告警。

---

## 阶段 2：配乐选曲与落轨

1. 读取 [01_选曲匹配与试听确认](references/music/01_选曲匹配与试听确认.md)，读精剪 `.srt`（纯配乐任务先向用户获取 `.srt` 或文案），按简洁模板推荐 3~5 首候选曲目、一句话原因、建议试听段落与大致位置，待用户确认。
2. 读取 [02_音乐剪辑3种模式](references/music/02_音乐剪辑3种模式.md) 与 [03_听感边界与工具参数](references/music/03_听感边界与工具参数.md)，将最终音乐分段与双音轨交叉淡入淡出写入 `timeline_edit.json` 的 `tracks.audio`。

**完成标准**：用户已确认曲目；`timeline_edit.json` 已包含最终 `tracks.audio`；`execute_timeline_edit.py --check` 零报错通过。

---

## 阶段 3：执行交付

读取 [01_时间线执行格式与交付](references/01_时间线执行格式与交付.md)，运行 `scripts/execute_timeline_edit.py`（4 选 1：`jianying` / `adudir` / `ffmpeg` / `plan`），按该文档中的**极简交付汇报模板**向用户只报告重点结果（成片概况、交付路径、配套文件，不贴冗长日志或整段 JSON）。

**完成标准**：执行命令返回码为 0；目标产物与 `操作指引与剪辑说明.md` 均真实存在；对话中已输出简洁重点汇报。
