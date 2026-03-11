package pcm

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------- BuildWAVHeader 单元测试 ----------

func TestBuildWAVHeader(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		dataSize      int
		sampleRate    int
		channels      int
		bitsPerSample int
	}{
		{
			name:          "标准16kHz单声道16bit",
			dataSize:      32000,
			sampleRate:    16000,
			channels:      1,
			bitsPerSample: 16,
		},
		{
			name:          "8kHz单声道16bit",
			dataSize:      16000,
			sampleRate:    8000,
			channels:      1,
			bitsPerSample: 16,
		},
		{
			name:          "44.1kHz立体声16bit",
			dataSize:      176400,
			sampleRate:    44100,
			channels:      2,
			bitsPerSample: 16,
		},
		{
			name:          "48kHz单声道24bit",
			dataSize:      144000,
			sampleRate:    48000,
			channels:      1,
			bitsPerSample: 24,
		},
		{
			name:          "零数据长度",
			dataSize:      0,
			sampleRate:    16000,
			channels:      1,
			bitsPerSample: 16,
		},
		{
			name:          "大数据长度",
			dataSize:      10 * 1024 * 1024, // 10MB
			sampleRate:    16000,
			channels:      1,
			bitsPerSample: 16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			h := BuildWAVHeader(tt.dataSize, tt.sampleRate, tt.channels, tt.bitsPerSample)

			// 头部总是44字节。
			require.Len(t, h, 44)

			// RIFF 标识。
			assert.Equal(t, "RIFF", string(h[0:4]))

			// 文件大小 = 36 + dataSize。
			fileSize := binary.LittleEndian.Uint32(h[4:8])
			assert.Equal(t, uint32(36+tt.dataSize), fileSize)

			// WAVE 标识。
			assert.Equal(t, "WAVE", string(h[8:12]))

			// fmt 子块。
			assert.Equal(t, "fmt ", string(h[12:16]))

			// fmt 块大小 = 16。
			fmtSize := binary.LittleEndian.Uint32(h[16:20])
			assert.Equal(t, uint32(16), fmtSize)

			// PCM 格式 = 1。
			audioFmt := binary.LittleEndian.Uint16(h[20:22])
			assert.Equal(t, uint16(1), audioFmt)

			// 声道数。
			ch := binary.LittleEndian.Uint16(h[22:24])
			assert.Equal(t, uint16(tt.channels), ch)

			// 采样率。
			sr := binary.LittleEndian.Uint32(h[24:28])
			assert.Equal(t, uint32(tt.sampleRate), sr)

			// 字节率 = sampleRate * channels * bitsPerSample / 8。
			expectedByteRate := uint32(tt.sampleRate * tt.channels * tt.bitsPerSample / 8)
			byteRate := binary.LittleEndian.Uint32(h[28:32])
			assert.Equal(t, expectedByteRate, byteRate)

			// 块对齐 = channels * bitsPerSample / 8。
			expectedBlockAlign := uint16(tt.channels * tt.bitsPerSample / 8)
			blockAlign := binary.LittleEndian.Uint16(h[32:34])
			assert.Equal(t, expectedBlockAlign, blockAlign)

			// 位深。
			bps := binary.LittleEndian.Uint16(h[34:36])
			assert.Equal(t, uint16(tt.bitsPerSample), bps)

			// data 子块标识。
			assert.Equal(t, "data", string(h[36:40]))

			// 数据大小。
			ds := binary.LittleEndian.Uint32(h[40:44])
			assert.Equal(t, uint32(tt.dataSize), ds)
		})
	}
}

// ---------- Fuzz 测试 ----------

func FuzzBuildWAVHeader(f *testing.F) {
	f.Add(32000, 16000, 1, 16)
	f.Add(0, 8000, 1, 16)
	f.Add(176400, 44100, 2, 16)
	f.Add(1, 1, 1, 8)

	f.Fuzz(func(t *testing.T, dataSize, sampleRate, channels, bitsPerSample int) {
		// 限制输入范围避免溢出。
		if dataSize < 0 || sampleRate < 0 || channels < 1 || channels > 8 || bitsPerSample < 8 || bitsPerSample > 32 {
			t.Skip("跳过无效参数")
		}

		h := BuildWAVHeader(dataSize, sampleRate, channels, bitsPerSample)

		// 必须返回44字节。
		require.Len(t, h, 44)
		// RIFF 和 WAVE 标识不变。
		assert.Equal(t, "RIFF", string(h[0:4]))
		assert.Equal(t, "WAVE", string(h[8:12]))
		assert.Equal(t, "fmt ", string(h[12:16]))
		assert.Equal(t, "data", string(h[36:40]))
	})
}

// ---------- 基准测试 ----------

func BenchmarkBuildWAVHeader(b *testing.B) {
	for range b.N {
		BuildWAVHeader(32000, 16000, 1, 16)
	}
}
