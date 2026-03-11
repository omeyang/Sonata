package pcm

import (
	"encoding/binary"
	"math"
)

// EnergyDBFS 计算 PCM16 LE 音频的能量（dBFS，相对满量程分贝）。
// 静音（全零或空输入）时返回 -96.0。
func EnergyDBFS(pcm []byte) float64 {
	if len(pcm) < 2 {
		return -96.0
	}

	sampleCount := len(pcm) / 2
	var sumSq float64

	for i := range sampleCount {
		s := float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
		sumSq += s * s
	}

	rms := math.Sqrt(sumSq / float64(sampleCount))
	if rms < 1.0 {
		return -96.0
	}

	// int16 的满量程为 32768。
	return 20.0 * math.Log10(rms/32768.0)
}

// FrameDuration 返回 PCM16 帧的时长（毫秒）。
func FrameDuration(pcm []byte, sampleRate int) int {
	if sampleRate <= 0 || len(pcm) < 2 {
		return 0
	}
	sampleCount := len(pcm) / 2
	return sampleCount * 1000 / sampleRate
}
