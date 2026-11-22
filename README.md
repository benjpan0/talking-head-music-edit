# talking-head-music-edit（口播精剪与视频配乐 Skill）

一个面向 AI Agent（Antigravity / WorkBuddy / Claude Code / Codex 等）的标准 Agent Skill，打通**口播逐字识别与精剪**、**友好单行字幕与纯净正文**、**音乐库匹配与双音轨配乐**，并通过统一的 `timeline_edit.json` 对接 **剪映专业版草稿 / ADu Director 工程 / FFmpeg 直接出片 / 仅输出剪辑方案**。

---

## 核心特性

- **三分支路由**：支持「只剪口播」「纯配乐 / 单曲剪辑」「口播 + 配乐」，按需加载规则文档。
- **初始化门禁与习惯记忆**：首次运行自动检查 Python / FFmpeg / 剪映真实草稿路径 / 逐字识别方式 / 音乐库目录，支持保存**常用词汇表（专有名词与纠错映射）**及**按视频类型的常用习惯**。
- **非机械式口播精剪**：
  - 结合逐字时间戳与双阈值物理静音（`-35dB` / `-28dB`），自动计算**有效语速（字/分）与加速建议**；
  - 提供 **3 种口播档位**（平衡型 / 专业紧凑型 / 人性化口播型）；
  - 顺畅多句合并为完整长片段，仅在重录、卡壳、废词或超长停顿处下刀，并自动校验防止逐句切碎；
  - 输出友好单行 `.srt` 字幕与 100% 纯净中文 `.txt` 正文。
- **音乐库自动整理与配乐落轨**：
  - 内置 `check_music_library.py` 自动检查/整理音乐库并补全 `music_catalog.json` 索引；
  - 支持 **两段式 / 三段式 / 停顿留白再续** 3 种配乐剪辑模式与**双音轨（`track 0 / 1`）交叉淡入淡出**。
- **统一中间执行格式（`timeline_edit.json`）与 4 选 1 交付**：
  - 剪映草稿（含双音轨淡入淡出与可选原生字幕轨）
  - ADu Director 工程（MEP 1 离线包或在线提交）
  - FFmpeg 直接渲染 MP4（可选烧录硬字幕）
  - 仅输出方案（附带可直接复制给智能体的 `操作指引与剪辑说明.md`）

---

## 目录结构

```text
talking-head-music-edit/
├── SKILL.md                      # Skill 入口：触发说明、3分支路由与阶段完成标准
├── README.md                     # 本说明文档
├── LICENSE                       # MIT 协议与第三方声明
├── requirements.txt              # Python 依赖清单
├── references/                   # 按需读取的流程与导演规则
│   ├── 00_初始化与素材转写.md
│   ├── 01_时间线执行格式与交付.md
│   ├── speech/                   # 口播档位、去重气口、字幕规则
│   └── music/                    # 选曲试听、3种模式、听感与工具参数
├── assets/                       # 配置模板、音乐库索引模板、时间线 Schema 与示例
└── scripts/                      # 确定性 Python 执行脚本与内置剪映草稿库
```

---

## 环境要求与安装

### 1. 基础环境
- **Python 3.10+**
- **FFmpeg / FFprobe**（需在 `PATH` 中可用）
- **NumPy**（仅音乐分析/渲染需要，见 `requirements.txt`）

```powershell
pip install -r requirements.txt
```

### 2. 安装为 Agent Skill
将整个 `talking-head-music-edit/` 文件夹复制到你的 AI 工具对应的 Skills 目录即可自动发现：
- Antigravity 工作区：`<项目根目录>/.agents/skills/talking-head-music-edit`
- Antigravity 全局：`~/.gemini/config/skills/talking-head-music-edit`
- 其他支持标准 `SKILL.md` 的 Agent（如 WorkBuddy / Claude Code / Codex）：按各自 Skills 目录放入即可。

---

## 音乐库文件夹规范

使用自动配乐功能前，准备一个本地音乐库目录（不随本仓库提供版权音乐），结构如下：

```text
<你的音乐库目录>/
├── audio/                  # 音乐源文件（.mp3 / .wav / .flac / .m4a / .aac）
├── analysis/               # 对应的毫秒制分析文件（<歌名>_剪辑数据.json）
└── music_catalog.json      # 快速配对索引（可由下方脚本自动生成）
```

运行检查或一键整理：
```powershell
python -B scripts/check_music_library.py --dir "<你的音乐库目录>" --check
python -B scripts/check_music_library.py --dir "<你的音乐库目录>" --organize
```

---

## 平台兼容与注意事项

- **平台支持**：剪映专业版草稿构建与首页登记目前在 **Windows** 环境实测通过；macOS / Linux 环境下建议优先使用 **FFmpeg 直接出片** 或 **ADu 工程** 交付。
- **云端语音识别**：如需使用云端 ASR（阿里云百炼 / 腾讯云 / Groq / OpenAI 兼容），请自行通过环境变量配置对应 API Key；本仓库不包含任何密钥，也支持完全离线的 `faster-whisper`。
- **剪映写入提示**：写入剪映草稿前请先关闭剪映专业版，避免首页索引文件并发冲突。

---

## 开源协议与第三方致谢

本项目主体采用 [MIT License](LICENSE) 开源。

`scripts/jy_vendor/` 内置并感谢以下第三方开源项目：
- **jy_vendor** by `luoluoluo22` — MIT License（见 `scripts/jy_vendor/LICENSE`）
- **[pyJianYingDraft](https://github.com/GuanYixuan/pyJianYingDraft)** by `GuanYixuan` — Apache License 2.0（见 `scripts/jy_vendor/vendor/pyJianYingDraft/LICENSE`）
