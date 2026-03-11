package pcm

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------- PCM16ToFloat32 单元测试 ----------

func TestPCM16ToFloat32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []byte
		want []float32
	}{
		{
			name: "nil输入返回nil",
			in:   nil,
			want: nil,
		},
		{
			name: "空切片返回nil",
			in:   []byte{},
			want: nil,
		},
		{
			name: "单字节不足返回nil",
			in:   []byte{0x01},
			want: nil,
		},
		{
			name: "静音(全零)返回0.0",
			in:   pcm16Bytes(0, 0),
			want: []float32{0.0, 0.0},
		},
		{
			name: "正满量程",
			in:   pcm16Bytes(32767),
			want: []float32{float32(32767) / 32768.0},
		},
		{
			name: "负满量程",
			in:   pcm16Bytes(-32768),
			want: []float32{-1.0},
		},
		{
			name: "单采样正值",
			in:   pcm16Bytes(16384),
			want: []float32{float32(16384) / 32768.0},
		},
		{
			name: "单采样负值",
			in:   pcm16Bytes(-16384),
			want: []float32{float32(-16384) / 32768.0},
		},
		{
			name: "多采样混合",
			in:   pcm16Bytes(0, 32767, -32768, 100),
			want: []float32{0.0, float32(32767) / 32768.0, -1.0, float32(100) / 32768.0},
		},
		{
			name: "奇数字节丢弃末尾",
			in:   append(pcm16Bytes(1000), 0xFF), // 3字节，只解析1个采样
			want: []float32{float32(1000) / 32768.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := PCM16ToFloat32(tt.in)
			if tt.want == nil {
				assert.Nil(t, got)
				return
			}
			require.Len(t, got, len(tt.want))
			for i := range tt.want {
				assert.InDelta(t, tt.want[i], got[i], 1e-6, "采样[%d]不一致", i)
			}
		})
	}
}

// ---------- Float32ToPCM16 单元测试 ----------

func TestFloat32ToPCM16(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []float32
		want []byte
	}{
		{
			name: "nil输入返回nil",
			in:   nil,
			want: nil,
		},
		{
			name: "空切片返回nil",
			in:   []float32{},
			want: nil,
		},
		{
			name: "零值",
			in:   []float32{0.0},
			want: pcm16Bytes(0),
		},
		{
			name: "正满量程钳位",
			in:   []float32{1.0},
			want: pcm16Bytes(32767),
		},
		{
			name: "负满量程",
			in:   []float32{-1.0},
			want: pcm16Bytes(-32767),
		},
		{
			name: "超出正范围被钳位",
			in:   []float32{2.5},
			want: pcm16Bytes(32767), // 钳位到1.0
		},
		{
			name: "超出负范围被钳位",
			in:   []float32{-3.0},
			want: pcm16Bytes(-32767), // 钳位到-1.0
		},
		{
			name: "多采样",
			in:   []float32{0.0, 0.5, -0.5},
			want: pcm16Bytes(0, 16383, -16383),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := Float32ToPCM16(tt.in)
			if tt.want == nil {
				assert.Nil(t, got)
				return
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

// ---------- 往返测试 ----------

func TestRoundTrip_PCM16_Float32(t *testing.T) {
	t.Parallel()

	// 从 PCM16 出发，转换到 float32 再转回，应几乎相等。
	original := pcm16Bytes(0, 100, -100, 32767, -32768, 16000, -16000)
	floats := PCM16ToFloat32(original)
	require.NotNil(t, floats)

	backToPCM := Float32ToPCM16(floats)
	require.NotNil(t, backToPCM)
	require.Len(t, backToPCM, len(original))

	// 由于浮点精度和乘除系数差异（32768 vs 32767），允许 ±1 的误差。
	for i := range len(original) / 2 {
		origSample := int16(uint16(original[i*2]) | uint16(original[i*2+1])<<8)
		gotSample := int16(uint16(backToPCM[i*2]) | uint16(backToPCM[i*2+1])<<8)
		diff := int(origSample) - int(gotSample)
		assert.LessOrEqual(t, int(math.Abs(float64(diff))), 2,
			"采样[%d]: 原始=%d 往返=%d 差异=%d", i, origSample, gotSample, diff)
	}
}

func TestRoundTrip_Float32_PCM16(t *testing.T) {
	t.Parallel()

	// 从 float32 出发，转换到 PCM16 再转回。
	original := []float32{0.0, 0.5, -0.5, 0.99, -0.99}
	pcmData := Float32ToPCM16(original)
	require.NotNil(t, pcmData)

	back := PCM16ToFloat32(pcmData)
	require.NotNil(t, back)
	require.Len(t, back, len(original))

	for i := range original {
		assert.InDelta(t, original[i], back[i], 0.001, "采样[%d]不一致", i)
	}
}

// ---------- Fuzz 测试 ----------

func FuzzPCM16ToFloat32(f *testing.F) {
	// 种子语料
	f.Add([]byte{})
	f.Add([]byte{0x00})
	f.Add(pcm16Bytes(0))
	f.Add(pcm16Bytes(32767))
	f.Add(pcm16Bytes(-32768))
	f.Add(pcm16Bytes(0, 32767, -32768))

	f.Fuzz(func(t *testing.T, data []byte) {
		result := PCM16ToFloat32(data)
		if len(data) < 2 {
			assert.Nil(t, result)
			return
		}
		// 所有输出值必须在 [-1.0, 1.0] 范围内。
		for i, v := range result {
			assert.GreaterOrEqual(t, v, float32(-1.0), "采样[%d]超出下界", i)
			assert.LessOrEqual(t, v, float32(1.0), "采样[%d]超出上界", i)
		}
	})
}

func FuzzFloat32ToPCM16(f *testing.F) {
	// 种子语料：使用字节编码 float32 切片不太方便，直接用单个 float32。
	f.Add(float32(0.0))
	f.Add(float32(1.0))
	f.Add(float32(-1.0))
	f.Add(float32(0.5))
	f.Add(float32(2.0))
	f.Add(float32(-2.0))

	f.Fuzz(func(t *testing.T, val float32) {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Skip("跳过 NaN/Inf")
		}
		result := Float32ToPCM16([]float32{val})
		require.NotNil(t, result)
		assert.Len(t, result, 2)
	})
}

// ---------- 基准测试 ----------

func BenchmarkPCM16ToFloat32(b *testing.B) {
	// 模拟 20ms@16kHz = 320 采样 = 640 字节。
	frame := silenceFrame(320)
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		PCM16ToFloat32(frame)
	}
}

func BenchmarkFloat32ToPCM16(b *testing.B) {
	samples := make([]float32, 320)
	for i := range samples {
		samples[i] = float32(i) / 320.0
	}
	b.SetBytes(int64(len(samples) * 4))
	b.ResetTimer()

	for range b.N {
		Float32ToPCM16(samples)
	}
}
