// Package pcm 提供实时语音处理的基础工具：
// PCM 重采样、能量计算、WAV 文件构建。
//
// 所有函数处理小端（Little-Endian）16-bit PCM 单声道数据。
package pcm

import "encoding/binary"

// Resample8to16 使用线性插值将 8 kHz PCM16 LE 音频转换为 16 kHz。
// 每个输入采样产生两个输出采样：原始值和插值中点。
// 最后一个采样被复制（没有下一个采样可用于插值）。
//
// 输入和输出均为小端 int16 字节切片。
func Resample8to16(in []byte) []byte {
	if len(in) < 2 {
		return nil
	}

	sampleCount := len(in) / 2
	out := make([]byte, sampleCount*4) // 2x samples, 2 bytes each

	for i := range sampleCount {
		s := int16(binary.LittleEndian.Uint16(in[i*2:]))

		var next int16
		if i+1 < sampleCount {
			next = int16(binary.LittleEndian.Uint16(in[(i+1)*2:]))
		} else {
			next = s
		}

		mid := int16((int32(s) + int32(next)) / 2)

		binary.LittleEndian.PutUint16(out[i*4:], uint16(s))
		binary.LittleEndian.PutUint16(out[i*4+2:], uint16(mid))
	}

	return out
}

// Resample8to16Into 使用提供的缓冲区将 8 kHz PCM16 LE 音频转换为 16 kHz。
// 算法与 Resample8to16 完全一致，但允许调用方复用 dst 缓冲区以避免分配。
// 如果 dst 容量不足，会重新分配。返回写入后的切片。
// 输入不足 2 字节时返回 nil。
func Resample8to16Into(dst, in []byte) []byte {
	if len(in) < 2 {
		return nil
	}

	sampleCount := len(in) / 2
	needed := sampleCount * 4 // 2x samples, 2 bytes each

	if cap(dst) >= needed {
		dst = dst[:needed]
	} else {
		dst = make([]byte, needed)
	}

	for i := range sampleCount {
		s := int16(binary.LittleEndian.Uint16(in[i*2:]))

		var next int16
		if i+1 < sampleCount {
			next = int16(binary.LittleEndian.Uint16(in[(i+1)*2:]))
		} else {
			next = s
		}

		mid := int16((int32(s) + int32(next)) / 2)

		binary.LittleEndian.PutUint16(dst[i*4:], uint16(s))
		binary.LittleEndian.PutUint16(dst[i*4+2:], uint16(mid))
	}

	return dst
}

// Resample16to8 通过抽取（保留每隔一个采样）将 16 kHz PCM16 LE 音频转换为 8 kHz。
func Resample16to8(in []byte) []byte {
	if len(in) < 4 {
		return nil
	}

	sampleCount := len(in) / 2
	outCount := sampleCount / 2
	out := make([]byte, outCount*2)

	for i := range outCount {
		copy(out[i*2:i*2+2], in[i*4:i*4+2])
	}

	return out
}
