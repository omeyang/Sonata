package pcm

import "encoding/binary"

// PCM16ToFloat32 将 PCM16 LE 字节切片转换为归一化 float32 采样。
// 输出范围 [-1.0, 1.0]，除以 32768.0（int16 满量程）。
// 输入长度不足 2 字节时返回 nil。
func PCM16ToFloat32(pcm []byte) []float32 {
	if len(pcm) < 2 {
		return nil
	}

	n := len(pcm) / 2
	out := make([]float32, n)

	for i := range n {
		s := int16(binary.LittleEndian.Uint16(pcm[i*2:]))
		out[i] = float32(s) / 32768.0
	}

	return out
}

// Float32ToPCM16 将归一化 float32 采样转换为 PCM16 LE 字节切片。
// 输入应在 [-1.0, 1.0] 范围内，超出部分会被钳位。
// 输入为空时返回 nil。
func Float32ToPCM16(samples []float32) []byte {
	if len(samples) == 0 {
		return nil
	}

	out := make([]byte, len(samples)*2)

	for i, s := range samples {
		// 钳位到 [-1.0, 1.0]。
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}

		v := int16(s * 32767.0)
		binary.LittleEndian.PutUint16(out[i*2:], uint16(v))
	}

	return out
}
