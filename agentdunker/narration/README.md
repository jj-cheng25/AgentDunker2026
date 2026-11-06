# agentdunker.narration（M6：自动解说 + Mossland 配音 + 自动对齐）

事件时间轴 → 带时间窗的中文解说稿 → Mossland 多版合成并用 ASR 回读选优 → 对齐到事件、压低原声 → `narration.json` + 人声轨 + 混音轨。

```bash
python -m agentdunker.narration --events events.json --video clip.mp4 -o out/narration.json [--preview out/preview.mp4]
# 输出：out/narration.json、out/narration_vo.wav（纯人声）、out/narration_mix.wav（人声 + 压低的原声）
```

常用参数：`--identity identity.json`（M3 名字）、`--bed`（另给要压低的原声）、`--bed-db -6 --duck-db -20`（原声平时/解说时增益，`--bed-db -inf` 去掉原声）、`--video-start`（时间轴 0 点在视频中的位置）、`--no-llm`（不调写稿模型，只用模板句）。
Python 调用：`from agentdunker.narration import run; run(events, video, out_json, ...)`。

## 输入（PLAN 的 events 合同，加几个可选字段）

```jsonc
{"duration": 64.8,
 "events": [{"t": 12.2, "type": "drive", "actors": [7], "conf": 0.9, "note": "可选：一句动作描述（英文或中文）"}],
 "shot": {"release_t": 58.0, "shooter_track": 7, "made": true, "xfg": 0.213, "points": 3},
 "players": {"7": {"name_zh": "凯文·杜兰特", "conf": 0.95}},          // 或用 --identity 传 M3 的 identity.json
 "sections": [{"start": 21.97, "end": 26.8, "kind": "bridge"},       // M7 的成片分段：live/bridge/replay/freeze/resume
              {"start": 52.8, "end": 58.0, "kind": "freeze"}]}
```

- 名字：只用 `conf ≥ 0.6` 且有中文名的球员；其他人一律用“持球人/队友/防守人”。
- 数字：只允许 `shot.xfg`、`shot.points`、`allowed_numbers` 里的数；比分、时间、命中率一律不编。
- 细节：空心/打板/碰框/扣篮/上篮必须有事件 `note` 支持；命中与否必须与 `shot.made` 一致，未给结果就不说进没进。
- `bridge` 段固定念“接下来，让我们用三维视角，回看这次进攻战术！”；`freeze` 段有 xFG 时写一条带官方 xFG 的句子。

## 输出 `narration.json`

`lines[]`：`start/end`（语音起止，时间轴秒）、`text`、`audio`（单句 wav）、`event_ref`、`window`、`tempo`、`in_window`、`track_cer`（整轨回读对齐到本句的 CER，验收用）、`cer/pinyin_cer/heard`（单句回读，诊断用）、`onset_ok`、`rewrites`、`dropped`。
`quality`：`tier`（A=全部在窗内、每句整轨 CER ≤5%、起音完整；B=有句子未达标但已出声；C=无句子）、`track_cer`、`track_heard` 等。任何一环失败都降级（模板句、删句），不抛错退出。

## 参照片人工修改 → 自动规则

| 参照片人工修改（redub `edits.json`） | 自动规则 | 代码 |
|---|---|---|
| 整组前后移 0.5–0.8 s | 句首放在事件时刻前 0.12 s | `script.LEAD` |
| “末句轻压留换气”、紧接上一句 0.15 s | 句间至少 0.15 s；可借下一句富余 ≤0.5 s，后句顺延 | `align.GAP/SPILL` |
| “拉开”长腔压到 0.8 倍 | 短句语速低于 3 字/秒时加快（≤1.25 倍） | `align.MIN_RATE` |
| 超时加速 1.2 倍 | 超窗先让模型压缩字数并重录，仍超再变速（≤1.2 倍，不变调，rubberband） | `align.place_all` |
| 删除“迎着协防” | 窗口 < 0.8 s 的事件不单独成句，时间让给前一句；改写后仍放不下就删句 | `script.MIN_WINDOW` |
| “队”字起音发虚，句首加 `[pause 0.3s]` 重录 | 每次请求前后都加 `[pause 0.3s]`；语音从文件开头就开始的版本判为起音风险 | `voice.LEAD_IN/TAIL_OUT` |
| “终结”的“结”没吐清，重录选句尾最饱满的一版 | ASR 回读首尾音节必须一致，句尾不能贴着文件末尾 | `voice._analyze` |
| （新）ASR 总听成同音字（强侧→墙侧） | 整轨回读不过的句子自动换词重录一次 | `align.retake` |

每句先合成 3 版（不过再加，最多 5–7 版），按 CER、首尾音节、起音、时长和窗口打分。期望时长按 5.4 字/秒加上 0.6 s 停顿；不加停顿余量时短句 8 版里 0 版合格，加上后 5/8 合格。
1–5 字的短句单独送 ASR 经常识别为空，所以选优时把这一版连播两遍再识别。

## 依赖与凭据

`ffmpeg`（带 rubberband 滤镜）、numpy、scipy、requests；可选 `pypinyin`（同音字容忍的 CER）、`librosa`（`compare_ref` 的音高统计）。
MOSS TTS/ASR：`MOSS_API_KEY` 或 `~/.config/moss/key`；写稿模型（Mosshub `gemini-3.8-flash`）：`MOSSHUB_API_KEY` 或 `~/.config/mosshub/key`。
音色 `ADK_VOICE_ID`（默认 Mossland `865098d7-…`），模型 `moss-tts-1.5-flash`（平台未提供带日期快照）；MOSS 并发 `ADK_MOSS_CONCURRENCY`（默认 3，上限钳在 3，超过会被 429 限流）。

缓存与写盘：take、ASR 结果、脚本和全部输出都先写临时文件再原子改名；读取缓存时会校验每个 take（RIFF 头声明长度与文件实际长度一致、ffmpeg 能解出样本），被杀进程或磁盘故障截断的 take 会被删掉并重新合成，损坏的 JSON 缓存当作未命中，不会让整次旁白失败。

## 复现与对比

```bash
python -m agentdunker.narration --events agentdunker.narration/fixtures/ref_film.events.json \
  --video <参照片> --video-start 6.0 --bed bench/clips/c01_hou_at_dal_kd_ref.mp4 -o ref/narration.json --preview ref/preview.mp4
python -m agentdunker.narration.compare_ref --ours ref/narration.json --ref-srt <参照片解说 srt> --ref-audio <参照片正片音轨> -o ref/compare.md
pytest agentdunker.narration/tests      # 离线规则测试，不调 API
```
