# Sonata

实时语音对话引擎核心库。

提供流式 ASR→LLM→TTS 管线编排、表驱动媒体状态机（FSM）、WebRTC VAD 和 barge-in 打断检测。约 1500 行代码（不含测试和 sherpa 子模块），零框架依赖，标准库优先。

## 架构

```
用户语音 → Transport → Session.eventLoop → ASR → DialogEngine(LLM) → TTS → Transport → 用户
                            ↕
                  mediafsm.FSM（听/说/打断/超时）
```

Session 是中央编排器，通过两个接口与上层解耦：

| 接口 | 位置 | 职责 |
|------|------|------|
| `Transport` | `engine/transport.go` | 音频输入输出抽象 |
| `aiface.DialogEngine` | `engine/aiface/dialog.go` | 业务对话逻辑注入 |

## 包结构

```
Sonata/
├── engine/              核心运行时引擎
│   ├── session.go       Session 编排器：事件循环、流式管线、barge-in、静默超时
│   ├── transport.go     Transport 接口定义
│   ├── event.go         事件类型
│   ├── metrics.go       度量接口 + NopMetrics
│   ├── aiface/          AI 能力接口：ASR / LLM / TTS / DialogEngine / Warmer / 结构化错误
│   ├── mediafsm/        表驱动媒体状态机，转换规则可配置
│   └── pcm/             音频工具：重采样、能量检测、WAV、WebRTC VAD、PCM↔Float32
└── sherpa/              sherpa-onnx 封装（独立 go.mod，CGO 依赖）
```

### engine — 会话运行时

`engine.New(cfg).Run(ctx)` 启动完整对话生命周期：

- **事件循环** — 单 goroutine select 驱动，处理音频帧、ASR 事件、TTS 播放完成、静默超时
- **流式管线** — ASR 最终结果 → DialogEngine 流式生成 → 逐句 TTS 合成 → 逐句 Transport 播放
- **Barge-in 打断** — 连续 10 帧语音（≈200ms）触发打断，取消 TTS context 并停止播放
- **填充词** — ASR 结果到 LLM 响应之间播放预合成填充词，感知延迟从 ~2s 降至 ~200ms
- **保护机制** — 最大会话时长、最大静默、首次静默超时，2 次超时自动挂断
- **Metrics** — 可选的度量收集接口（ASR/LLM/TTS 延迟、barge-in 次数等）

关键接口：

```go
// Transport — 音频 IO 抽象
type Transport interface {
    AudioIn() <-chan []byte
    PlayAudio(ctx context.Context, audio []byte) error
    StopPlayback(ctx context.Context) error
    Close() error
}

// SpeechDetector — 可插拔的人声检测（WebRTC VAD / Silero VAD / 能量阈值退回）
type SpeechDetector interface {
    IsSpeech(frame []byte) (bool, error)
}
```

### aiface — AI 能力接口

流式优先的 provider 接口定义：

| 接口 | 方法 | 说明 |
|------|------|------|
| `ASRProvider` | `StartStream(ctx, cfg) → ASRStream` | 创建流式语音识别会话 |
| `LLMProvider` | `GenerateStream(ctx, msgs, cfg) → <-chan string` | 流式文本生成 |
| `TTSProvider` | `SynthesizeStream(ctx, textCh, cfg) → <-chan []byte` | 流式语音合成 |
| `DialogEngine` | `Opening()` / `ProcessStream(ctx, text)` / `Finished()` | 对话逻辑编排 |
| `Warmer` | `Warmup(ctx)` | 连接预热，避免首次调用冷启动 |

结构化错误 `aiface.Error` 区分临时/永久/超时错误，支持 `errors.As` 错误链提取。

### mediafsm — 媒体状态机

表驱动 FSM，转换规则作为 `[]Transition` 数据传入：

```go
fsm := mediafsm.NewFSM(mediafsm.Idle, mediafsm.AppTransitions())
fsm.OnTransition(func(from, to mediafsm.State, event mediafsm.Event) {
    log.Printf("%s → %s (on %s)", from, to, event)
})
fsm.Handle(mediafsm.EvAnswer)
```

预置两套转换表：
- **`PhoneTransitions()`** — Idle → Dialing → Ringing → AMD → BotSpeaking ↔ UserSpeaking → Hangup
- **`AppTransitions()`** — Idle → BotSpeaking ↔ UserSpeaking → Hangup（无拨号/振铃/AMD）

也支持自定义 `[]Transition` 构造任意状态流转。

完整状态集：`Idle`、`Dialing`、`Ringing`、`AMDDetecting`、`BotSpeaking`、`WaitingUser`、`UserSpeaking`、`Processing`、`BargeIn`、`SilenceTimeout`、`Hangup`、`PostProcessing`。

### pcm — 音频工具

底层音频处理，除 WebRTC VAD 外无外部依赖：

| 函数 | 说明 |
|------|------|
| `Resample8to16` / `Resample16to8` | 线性插值/抽取重采样 |
| `EnergyDBFS` | PCM16 LE 能量计算（dBFS） |
| `BuildWAVHeader` | 44 字节标准 WAV 头 |
| `PCM16ToFloat32` / `Float32ToPCM16` | PCM16 LE ↔ float32 格式转换 |
| `NewVAD` | WebRTC VAD 封装，4 级灵敏度 |

### sherpa — 端侧推理（独立模块）

独立 Go 模块（`sherpa/go.mod`），封装 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 提供端侧 ASR/TTS/VAD 能力。有 CGO 依赖，需要单独构建。

- **SileroVAD** — 基于 Silero 模型的 VAD（实现 `SpeechDetector` 接口）
- **ASR / RaceASR** — 端侧语音识别，RaceASR 支持多引擎竞速
- **TTS / TieredTTS** — 端侧语音合成，支持分级降级

## 快速开始

```bash
go get github.com/omeyang/Sonata
```

```go
import (
    "github.com/omeyang/Sonata/engine"
    "github.com/omeyang/Sonata/engine/mediafsm"
)

cfg := engine.Config{
    SessionID:   "session-001",
    Transport:   myTransport,                    // 实现 engine.Transport
    Dialogue:    myDialogue,                     // 实现 aiface.DialogEngine
    ASR:         myASR,                          // 实现 aiface.ASRProvider
    TTS:         myTTS,                          // 实现 aiface.TTSProvider
    Transitions: mediafsm.AppTransitions(),
    Protection: engine.ProtectionConfig{
        MaxDurationSec:         300,
        MaxSilenceSec:          15,
        FirstSilenceTimeoutSec: 6,
    },
}

s := engine.New(cfg)
s.FSM().Handle(mediafsm.EvAnswer)
result, err := s.Run(ctx)
```

## 构建与测试

使用 [go-task](https://taskfile.dev/)（见 `Taskfile.yml`）：

```bash
task test              # 运行所有测试
task test:unit         # 仅单元测试（-short）
task test:bench        # 基准测试
task lint              # golangci-lint
task cover             # 覆盖率报告
task ci                # CI 流水线（lint + test + cover）

# 单个测试
go test ./engine/pcm/ -run TestResample8to16

# sherpa 子模块（需要 CGO）
task sherpa:test
```

## 技术栈

- Go 1.25+
- [go-webrtcvad](https://github.com/maxhawkins/go-webrtcvad) — WebRTC VAD 绑定
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（sherpa 子模块，可选）
- 标准库优先，零框架依赖

## 许可证

[AGPL-3.0](LICENSE)
