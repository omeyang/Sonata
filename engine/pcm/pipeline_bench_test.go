package pcm

import (
	"testing"
)

// BenchmarkAudioPipeline 基准测试完整音频帧处理管线：
// 重采样(8→16kHz) + 能量检测 + VAD。
// 这是 Session.handleAudioFrame 热路径的核心计算。
func BenchmarkAudioPipeline(b *testing.B) {
	frame8k := loudFrame(160, 5000) // 20ms@8kHz = 320 bytes
	vad, err := NewVAD(8000, 3)
	if err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(frame8k)))
	b.ResetTimer()

	for range b.N {
		resampled := Resample8to16(frame8k)
		_ = EnergyDBFS(frame8k)
		_, _ = vad.IsSpeech(frame8k)
		_ = resampled
	}
}

// BenchmarkAudioPipelineNoResample 无需重采样时的管线（16kHz 输入）。
func BenchmarkAudioPipelineNoResample(b *testing.B) {
	frame16k := loudFrame(320, 5000) // 20ms@16kHz = 640 bytes
	vad, err := NewVAD(16000, 3)
	if err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(frame16k)))
	b.ResetTimer()

	for range b.N {
		_ = EnergyDBFS(frame16k)
		_, _ = vad.IsSpeech(frame16k)
	}
}

// BenchmarkResample8to16FrameSizes 不同帧大小下的重采样性能。
func BenchmarkResample8to16FrameSizes(b *testing.B) {
	sizes := []struct {
		name    string
		samples int // 8kHz 采样数
	}{
		{"10ms_80samp", 80},
		{"20ms_160samp", 160},
		{"30ms_240samp", 240},
	}

	for _, sz := range sizes {
		frame := loudFrame(sz.samples, 5000)
		b.Run(sz.name, func(b *testing.B) {
			b.SetBytes(int64(len(frame)))
			for range b.N {
				Resample8to16(frame)
			}
		})
	}
}

// BenchmarkEnergyDBFSFrameSizes 不同帧大小下的能量检测性能。
func BenchmarkEnergyDBFSFrameSizes(b *testing.B) {
	sizes := []struct {
		name    string
		samples int
	}{
		{"10ms_8k", 80},
		{"20ms_8k", 160},
		{"30ms_8k", 240},
		{"20ms_16k", 320},
	}

	for _, sz := range sizes {
		frame := loudFrame(sz.samples, 5000)
		b.Run(sz.name, func(b *testing.B) {
			b.SetBytes(int64(len(frame)))
			for range b.N {
				EnergyDBFS(frame)
			}
		})
	}
}

// BenchmarkPCM16ToFloat32AndBack 基准测试 PCM16→Float32→PCM16 往返转换。
// 这是 sherpa ASR 送帧的典型路径。
func BenchmarkPCM16ToFloat32AndBack(b *testing.B) {
	frame := loudFrame(320, 5000) // 20ms@16kHz

	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		f32 := PCM16ToFloat32(frame)
		_ = Float32ToPCM16(f32)
	}
}

// BenchmarkResampleAlloc 测量重采样在高频调用下的分配压力。
// 模拟 1 秒@8kHz（50 帧×20ms）的累计分配。
func BenchmarkResampleAlloc(b *testing.B) {
	frames := make([][]byte, 50)
	for i := range frames {
		frames[i] = loudFrame(160, int16(i*100))
	}

	b.ResetTimer()
	b.ReportAllocs()

	for range b.N {
		for _, f := range frames {
			_ = Resample8to16(f)
		}
	}
}
