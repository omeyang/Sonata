package engine

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestNopMetricsImplementsMetrics 验证 NopMetrics 实现 Metrics 接口。
func TestNopMetricsImplementsMetrics(t *testing.T) {
	t.Parallel()

	var m Metrics = NopMetrics{}
	assert.NotNil(t, m, "NopMetrics 应实现 Metrics 接口")
}

// TestNopMetricsAllMethods 验证 NopMetrics 所有方法可调用且不 panic。
func TestNopMetricsAllMethods(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		fn   func(NopMetrics)
	}{
		{"RecordASRLatency", func(m NopMetrics) { m.RecordASRLatency(100 * time.Millisecond) }},
		{"RecordLLMFirstToken", func(m NopMetrics) { m.RecordLLMFirstToken(200 * time.Millisecond) }},
		{"RecordTTSFirstChunk", func(m NopMetrics) { m.RecordTTSFirstChunk(150 * time.Millisecond) }},
		{"RecordTurnLatency", func(m NopMetrics) { m.RecordTurnLatency(500 * time.Millisecond) }},
		{"IncBargeIn", func(m NopMetrics) { m.IncBargeIn() }},
		{"IncSilenceTimeout", func(m NopMetrics) { m.IncSilenceTimeout() }},
		{"IncProviderError", func(m NopMetrics) { m.IncProviderError("test_provider") }},
		{"SetCallActive", func(m NopMetrics) { m.SetCallActive(5) }},
		{"IncFillerPlayed", func(m NopMetrics) { m.IncFillerPlayed() }},
		{"IncSpeculativeHit", func(m NopMetrics) { m.IncSpeculativeHit() }},
		{"IncSpeculativeMiss", func(m NopMetrics) { m.IncSpeculativeMiss() }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.NotPanics(t, func() { tt.fn(NopMetrics{}) })
		})
	}
}

// TestNopMetricsZeroValue 验证零值 NopMetrics 可直接使用。
func TestNopMetricsZeroValue(t *testing.T) {
	t.Parallel()

	var m NopMetrics
	assert.NotPanics(t, func() {
		m.RecordASRLatency(0)
		m.RecordLLMFirstToken(0)
		m.RecordTTSFirstChunk(0)
		m.RecordTurnLatency(0)
		m.IncBargeIn()
		m.IncSilenceTimeout()
		m.IncProviderError("")
		m.SetCallActive(0)
		m.IncFillerPlayed()
		m.IncSpeculativeHit()
		m.IncSpeculativeMiss()
	})
}

// TestNopMetricsMultipleCalls 验证多次调用不会累积或 panic。
func TestNopMetricsMultipleCalls(t *testing.T) {
	t.Parallel()

	m := NopMetrics{}
	for range 100 {
		m.IncBargeIn()
		m.IncFillerPlayed()
		m.RecordASRLatency(time.Millisecond)
	}
}

// TestNopMetricsEdgeCases 验证边界参数不会 panic。
func TestNopMetricsEdgeCases(t *testing.T) {
	t.Parallel()

	m := NopMetrics{}

	// 零值和极端值。
	m.RecordASRLatency(0)
	m.RecordASRLatency(-1 * time.Second)
	m.RecordASRLatency(24 * time.Hour)
	m.RecordLLMFirstToken(0)
	m.RecordTTSFirstChunk(0)
	m.RecordTurnLatency(0)
	m.IncProviderError("")
	m.IncProviderError("中文提供者")
	m.SetCallActive(0)
	m.SetCallActive(-1)
	m.SetCallActive(999999)
}

// ---------------------------------------------------------------------------
// 基准测试
// ---------------------------------------------------------------------------

// BenchmarkNopMetricsRecordASRLatency 基准测试延迟记录。
func BenchmarkNopMetricsRecordASRLatency(b *testing.B) {
	m := NopMetrics{}
	d := 100 * time.Millisecond
	b.ResetTimer()
	for range b.N {
		m.RecordASRLatency(d)
	}
}

// BenchmarkNopMetricsIncBargeIn 基准测试计数操作。
func BenchmarkNopMetricsIncBargeIn(b *testing.B) {
	m := NopMetrics{}
	b.ResetTimer()
	for range b.N {
		m.IncBargeIn()
	}
}
