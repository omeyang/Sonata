package engine

import "time"

// Metrics 是会话级别的度量收集接口。
// 实现方可对接 Prometheus、OpenTelemetry 或其他后端。
// 传入 nil 时所有调用静默跳过（零值安全）。
type Metrics interface {
	// RecordASRLatency 记录 ASR 识别延迟。
	RecordASRLatency(duration time.Duration)

	// RecordLLMFirstToken 记录 LLM 首 token 延迟。
	RecordLLMFirstToken(duration time.Duration)

	// RecordTTSFirstChunk 记录 TTS 首包延迟。
	RecordTTSFirstChunk(duration time.Duration)

	// RecordTurnLatency 记录完整轮次延迟（用户说完 -> 机器人开始回复）。
	RecordTurnLatency(duration time.Duration)

	// IncBargeIn 记录一次 barge-in 事件。
	IncBargeIn()

	// IncSilenceTimeout 记录一次静默超时事件。
	IncSilenceTimeout()

	// IncProviderError 记录一次提供者错误。
	IncProviderError(provider string)

	// SetCallActive 设置活跃通话数量（用于 gauge）。
	SetCallActive(count int)

	// IncFillerPlayed 记录一次填充词播放事件。
	IncFillerPlayed()

	// IncSpeculativeHit 记录一次预推理命中。
	IncSpeculativeHit()

	// IncSpeculativeMiss 记录一次预推理未命中。
	IncSpeculativeMiss()
}

// NopMetrics 是不执行任何操作的 Metrics 实现（默认值）。
type NopMetrics struct{}

// RecordASRLatency 实现 Metrics 接口（空操作）.
func (NopMetrics) RecordASRLatency(time.Duration) {}

// RecordLLMFirstToken 实现 Metrics 接口（空操作）.
func (NopMetrics) RecordLLMFirstToken(time.Duration) {}

// RecordTTSFirstChunk 实现 Metrics 接口（空操作）.
func (NopMetrics) RecordTTSFirstChunk(time.Duration) {}

// RecordTurnLatency 实现 Metrics 接口（空操作）.
func (NopMetrics) RecordTurnLatency(time.Duration) {}

// IncBargeIn 实现 Metrics 接口（空操作）.
func (NopMetrics) IncBargeIn() {}

// IncSilenceTimeout 实现 Metrics 接口（空操作）.
func (NopMetrics) IncSilenceTimeout() {}

// IncProviderError 实现 Metrics 接口（空操作）.
func (NopMetrics) IncProviderError(string) {}

// SetCallActive 实现 Metrics 接口（空操作）.
func (NopMetrics) SetCallActive(int) {}

// IncFillerPlayed 实现 Metrics 接口（空操作）.
func (NopMetrics) IncFillerPlayed() {}

// IncSpeculativeHit 实现 Metrics 接口（空操作）.
func (NopMetrics) IncSpeculativeHit() {}

// IncSpeculativeMiss 实现 Metrics 接口（空操作）.
func (NopMetrics) IncSpeculativeMiss() {}
