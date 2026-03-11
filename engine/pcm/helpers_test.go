package pcm

import "encoding/binary"

// silenceFrame 生成指定采样数的静音帧（全零 PCM16 LE）。
func silenceFrame(samples int) []byte {
	return make([]byte, samples*2)
}

// loudFrame 生成指定采样数的大音量帧（所有采样设为 amplitude）。
// amplitude 范围：[-32768, 32767]。
func loudFrame(samples int, amplitude int16) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(amplitude))
	}
	return buf
}

// sineFrame 生成简单的交替正负采样帧，用于模拟有能量的音频。
func sineFrame(samples int) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		var v int16
		if i%2 == 0 {
			v = 16384
		} else {
			v = -16384
		}
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	return buf
}

// pcm16Bytes 将 int16 切片编码为 PCM16 LE 字节切片。
func pcm16Bytes(samples ...int16) []byte {
	buf := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(s))
	}
	return buf
}
