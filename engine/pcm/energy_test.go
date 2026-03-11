package pcm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// ---------- EnergyDBFS 单元测试 ----------

func TestEnergyDBFS(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		in     []byte
		expect float64
		delta  float64 // 允许误差
	}{
		{
			name:   "nil输入返回-96",
			in:     nil,
			expect: -96.0,
			delta:  0,
		},
		{
			name:   "空切片返回-96",
			in:     []byte{},
			expect: -96.0,
			delta:  0,
		},
		{
			name:   "单字节不足返回-96",
			in:     []byte{0xFF},
			expect: -96.0,
			delta:  0,
		},
		{
			name:   "静音帧返回-96",
			in:     silenceFrame(160),
			expect: -96.0,
			delta:  0,
		},
		{
			name:   "满量程正弦波约0dBFS",
			in:     loudFrame(160, 32767),
			expect: 20.0 * math.Log10(32767.0/32768.0),
			delta:  0.01,
		},
		{
			name:   "满量程负值约0dBFS",
			in:     loudFrame(160, -32768),
			expect: 0.0, // RMS = 32768, 20*log10(32768/32768) = 0
			delta:  0.01,
		},
		{
			name:   "半量程约-6dBFS",
			in:     loudFrame(160, 16384),
			expect: 20.0 * math.Log10(16384.0/32768.0),
			delta:  0.01,
		},
		{
			name:   "极小值(1)_RMS等于1不低于1",
			in:     pcm16Bytes(1),
			expect: 20.0 * math.Log10(1.0/32768.0), // RMS=1.0，不触发 <1 分支
			delta:  0.01,
		},
		{
			name:   "奇数字节丢弃末尾",
			in:     append(loudFrame(1, 10000), 0xFF),
			expect: 20.0 * math.Log10(10000.0/32768.0),
			delta:  0.01,
		},
		{
			name:   "混合正负采样",
			in:     sineFrame(160),
			expect: 20.0 * math.Log10(16384.0/32768.0),
			delta:  0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := EnergyDBFS(tt.in)
			assert.InDelta(t, tt.expect, got, tt.delta, "能量值不匹配")
		})
	}
}

// ---------- FrameDuration 单元测试 ----------

func TestFrameDuration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		pcm        []byte
		sampleRate int
		want       int
	}{
		{
			name:       "nil输入返回0",
			pcm:        nil,
			sampleRate: 16000,
			want:       0,
		},
		{
			name:       "空输入返回0",
			pcm:        []byte{},
			sampleRate: 16000,
			want:       0,
		},
		{
			name:       "单字节不足返回0",
			pcm:        []byte{0x00},
			sampleRate: 16000,
			want:       0,
		},
		{
			name:       "采样率为零返回0",
			pcm:        silenceFrame(160),
			sampleRate: 0,
			want:       0,
		},
		{
			name:       "负采样率返回0",
			pcm:        silenceFrame(160),
			sampleRate: -1,
			want:       0,
		},
		{
			name:       "16kHz_160采样=10ms",
			pcm:        silenceFrame(160),
			sampleRate: 16000,
			want:       10,
		},
		{
			name:       "16kHz_320采样=20ms",
			pcm:        silenceFrame(320),
			sampleRate: 16000,
			want:       20,
		},
		{
			name:       "8kHz_80采样=10ms",
			pcm:        silenceFrame(80),
			sampleRate: 8000,
			want:       10,
		},
		{
			name:       "8kHz_240采样=30ms",
			pcm:        silenceFrame(240),
			sampleRate: 8000,
			want:       30,
		},
		{
			name:       "32kHz_320采样=10ms",
			pcm:        silenceFrame(320),
			sampleRate: 32000,
			want:       10,
		},
		{
			name:       "奇数字节丢弃末尾",
			pcm:        append(silenceFrame(160), 0x00), // 321字节 → 160采样
			sampleRate: 16000,
			want:       10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := FrameDuration(tt.pcm, tt.sampleRate)
			assert.Equal(t, tt.want, got)
		})
	}
}

// ---------- Fuzz 测试 ----------

func FuzzEnergyDBFS(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00})
	f.Add(silenceFrame(10))
	f.Add(loudFrame(10, 32767))
	f.Add(pcm16Bytes(-32768))

	f.Fuzz(func(t *testing.T, data []byte) {
		result := EnergyDBFS(data)
		// 结果必须在 [-96, 0] 范围内。
		assert.GreaterOrEqual(t, result, -96.0)
		assert.LessOrEqual(t, result, 0.01) // 浮点容差
	})
}

// ---------- 基准测试 ----------

func BenchmarkEnergyDBFS(b *testing.B) {
	frame := sineFrame(320) // 20ms@16kHz
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		EnergyDBFS(frame)
	}
}

func BenchmarkFrameDuration(b *testing.B) {
	frame := silenceFrame(320)
	b.ResetTimer()

	for range b.N {
		FrameDuration(frame, 16000)
	}
}
