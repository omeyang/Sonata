package pcm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------- Resample8to16 单元测试 ----------

func TestResample8to16(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []byte
		want []byte
	}{
		{
			name: "nil输入返回nil",
			in:   nil,
			want: nil,
		},
		{
			name: "空输入返回nil",
			in:   []byte{},
			want: nil,
		},
		{
			name: "单字节不足返回nil",
			in:   []byte{0x01},
			want: nil,
		},
		{
			name: "单采样复制(无插值邻居)",
			in:   pcm16Bytes(1000),
			// 最后一个采样：mid = (1000+1000)/2 = 1000
			want: pcm16Bytes(1000, 1000),
		},
		{
			name: "两采样线性插值",
			in:   pcm16Bytes(0, 100),
			// 第一采样：s=0, next=100, mid=50 → [0, 50]
			// 第二采样：s=100, next=100(最后复制), mid=100 → [100, 100]
			want: pcm16Bytes(0, 50, 100, 100),
		},
		{
			name: "负值采样",
			in:   pcm16Bytes(-1000, -2000),
			// 第一采样：mid=(-1000+-2000)/2=-1500
			// 第二采样：mid=(-2000+-2000)/2=-2000
			want: pcm16Bytes(-1000, -1500, -2000, -2000),
		},
		{
			name: "满量程边界",
			in:   pcm16Bytes(32767, -32768),
			// mid = (32767 + (-32768)) / 2 = -1/2 = 0 (整数除法)
			want: pcm16Bytes(32767, 0, -32768, -32768),
		},
		{
			name: "三采样",
			in:   pcm16Bytes(0, 1000, 2000),
			want: pcm16Bytes(0, 500, 1000, 1500, 2000, 2000),
		},
		{
			name: "奇数字节丢弃末尾",
			in:   append(pcm16Bytes(500), 0xFF), // 3字节 → 只解析1个采样
			want: pcm16Bytes(500, 500),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := Resample8to16(tt.in)
			if tt.want == nil {
				assert.Nil(t, got)
				return
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

// ---------- Resample16to8 单元测试 ----------

func TestResample16to8(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []byte
		want []byte
	}{
		{
			name: "nil输入返回nil",
			in:   nil,
			want: nil,
		},
		{
			name: "空输入返回nil",
			in:   []byte{},
			want: nil,
		},
		{
			name: "单字节不足返回nil",
			in:   []byte{0x01},
			want: nil,
		},
		{
			name: "两字节不足返回nil(需要至少4字节)",
			in:   pcm16Bytes(100),
			want: nil,
		},
		{
			name: "三字节不足返回nil",
			in:   []byte{0x00, 0x01, 0x02},
			want: nil,
		},
		{
			name: "两采样抽取第一个",
			in:   pcm16Bytes(1000, 2000),
			want: pcm16Bytes(1000),
		},
		{
			name: "四采样抽取偶数位",
			in:   pcm16Bytes(100, 200, 300, 400),
			want: pcm16Bytes(100, 300),
		},
		{
			name: "奇数采样数丢弃最后一个",
			in:   pcm16Bytes(100, 200, 300, 400, 500), // 5采样
			// sampleCount=5, outCount=2
			want: pcm16Bytes(100, 300),
		},
		{
			name: "负值采样",
			in:   pcm16Bytes(-1000, -2000, -3000, -4000),
			want: pcm16Bytes(-1000, -3000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := Resample16to8(tt.in)
			if tt.want == nil {
				assert.Nil(t, got)
				return
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

// ---------- Resample8to16Into 单元测试 ----------

func TestResample8to16Into(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []byte
		want []byte
	}{
		{
			name: "nil输入返回nil",
			in:   nil,
			want: nil,
		},
		{
			name: "单字节不足返回nil",
			in:   []byte{0x01},
			want: nil,
		},
		{
			name: "两采样线性插值",
			in:   pcm16Bytes(0, 100),
			want: pcm16Bytes(0, 50, 100, 100),
		},
		{
			name: "三采样",
			in:   pcm16Bytes(0, 1000, 2000),
			want: pcm16Bytes(0, 500, 1000, 1500, 2000, 2000),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := Resample8to16Into(nil, tt.in)
			if tt.want == nil {
				assert.Nil(t, got)
				return
			}
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestResample8to16Into_Consistency 验证 Into 与原版输出一致。
func TestResample8to16Into_Consistency(t *testing.T) {
	t.Parallel()

	frames := [][]byte{
		pcm16Bytes(100, 200, 300),
		silenceFrame(160),
		loudFrame(160, 10000),
		sineFrame(80),
	}
	for _, frame := range frames {
		want := Resample8to16(frame)
		got := Resample8to16Into(nil, frame)
		assert.Equal(t, want, got)
	}
}

// TestResample8to16Into_BufferReuse 验证缓冲区复用不影响正确性。
func TestResample8to16Into_BufferReuse(t *testing.T) {
	t.Parallel()

	buf := make([]byte, 0, 640) // 预分配
	frame1 := pcm16Bytes(100, 200)
	frame2 := pcm16Bytes(300, 400)

	// 第一次调用。
	buf = Resample8to16Into(buf, frame1)
	assert.Equal(t, Resample8to16(frame1), buf)

	// 第二次复用同一个 buf，内容完全覆写。
	buf = Resample8to16Into(buf, frame2)
	assert.Equal(t, Resample8to16(frame2), buf)
}

// TestResample8to16Into_BufferGrowth 验证容量不足时自动扩容。
func TestResample8to16Into_BufferGrowth(t *testing.T) {
	t.Parallel()

	smallBuf := make([]byte, 0, 4) // 容量极小
	frame := pcm16Bytes(100, 200, 300)
	result := Resample8to16Into(smallBuf, frame)

	assert.Equal(t, Resample8to16(frame), result)
	assert.GreaterOrEqual(t, cap(result), 12) // 3 采样 * 4 字节/采样 = 12 字节
}

// ---------- 往返测试：8→16→8 ----------

func TestResampleRoundTrip_8to16to8(t *testing.T) {
	t.Parallel()

	original := pcm16Bytes(100, 200, 300, 400, 500)
	upsampled := Resample8to16(original)
	require.NotNil(t, upsampled)

	downsampled := Resample16to8(upsampled)
	require.NotNil(t, downsampled)

	// 8→16 每采样产生2个，再16→8抽取偶数位，应恢复原始数据。
	assert.Equal(t, original, downsampled)
}

// ---------- 输出长度验证 ----------

func TestResample8to16_OutputLength(t *testing.T) {
	t.Parallel()

	// N个采样输入，应产生2N个采样输出（4N字节）。
	for _, n := range []int{1, 2, 10, 80, 160} {
		in := silenceFrame(n)
		out := Resample8to16(in)
		require.NotNil(t, out, "输入%d采样不应返回nil", n)
		assert.Len(t, out, n*4, "输入%d采样输出长度不对", n)
	}
}

func TestResample16to8_OutputLength(t *testing.T) {
	t.Parallel()

	// 2N个采样输入，应产生N个采样输出（2N字节）。
	for _, n := range []int{2, 4, 10, 80, 160, 320} {
		in := silenceFrame(n)
		out := Resample16to8(in)
		require.NotNil(t, out, "输入%d采样不应返回nil", n)
		assert.Len(t, out, n, "输入%d采样输出长度不对", n) // n samples → n/2 samples → n bytes
	}
}

// ---------- Fuzz 测试 ----------

func FuzzResample8to16(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00})
	f.Add(pcm16Bytes(0))
	f.Add(pcm16Bytes(32767, -32768))
	f.Add(silenceFrame(80))

	f.Fuzz(func(t *testing.T, data []byte) {
		result := Resample8to16(data)
		if len(data) < 2 {
			assert.Nil(t, result)
			return
		}
		// 输出长度应为输入采样数*4字节。
		sampleCount := len(data) / 2
		assert.Len(t, result, sampleCount*4)
	})
}

func FuzzResample16to8(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x00, 0x01})
	f.Add(pcm16Bytes(1000, 2000))
	f.Add(silenceFrame(160))

	f.Fuzz(func(t *testing.T, data []byte) {
		result := Resample16to8(data)
		if len(data) < 4 {
			assert.Nil(t, result)
			return
		}
		sampleCount := len(data) / 2
		outCount := sampleCount / 2
		assert.Len(t, result, outCount*2)
	})
}

// ---------- 基准测试 ----------

func BenchmarkResample8to16(b *testing.B) {
	// 模拟 20ms@8kHz = 160 采样 = 320 字节。
	frame := silenceFrame(160)
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		Resample8to16(frame)
	}
}

func BenchmarkResample8to16Into(b *testing.B) {
	frame := silenceFrame(160)
	buf := make([]byte, 0, 640)
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		buf = Resample8to16Into(buf, frame)
	}
}

func BenchmarkResample16to8(b *testing.B) {
	// 模拟 20ms@16kHz = 320 采样 = 640 字节。
	frame := silenceFrame(320)
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		Resample16to8(frame)
	}
}
