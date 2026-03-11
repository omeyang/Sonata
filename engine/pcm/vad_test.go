package pcm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------- NewVAD 单元测试 ----------

func TestNewVAD(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		sampleRate int
		mode       VADMode
		wantErr    bool
	}{
		{
			name:       "16kHz_Quality模式",
			sampleRate: 16000,
			mode:       VADQuality,
			wantErr:    false,
		},
		{
			name:       "8kHz_LowBitrate模式",
			sampleRate: 8000,
			mode:       VADLowBitrate,
			wantErr:    false,
		},
		{
			name:       "16kHz_Aggressive模式",
			sampleRate: 16000,
			mode:       VADAggressive,
			wantErr:    false,
		},
		{
			name:       "16kHz_VeryAggressive模式",
			sampleRate: 16000,
			mode:       VADVeryAggressive,
			wantErr:    false,
		},
		{
			name:       "32kHz_Quality模式",
			sampleRate: 32000,
			mode:       VADQuality,
			wantErr:    false,
		},
		{
			name:       "无效mode(-1)",
			sampleRate: 16000,
			mode:       VADMode(-1),
			wantErr:    true,
		},
		{
			name:       "无效mode(4)",
			sampleRate: 16000,
			mode:       VADMode(4),
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			vad, err := NewVAD(tt.sampleRate, tt.mode)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, vad)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, vad)
		})
	}
}

// ---------- IsSpeech 单元测试 ----------

func TestVAD_IsSpeech(t *testing.T) {
	t.Parallel()

	// 创建 16kHz VAD 实例。
	vad, err := NewVAD(16000, VADVeryAggressive)
	require.NoError(t, err)

	tests := []struct {
		name    string
		frame   []byte
		wantErr bool
		// 不验证 active 具体值，因为 VAD 检测结果依赖算法内部状态。
	}{
		{
			name:    "静音帧20ms(640字节)无错误",
			frame:   silenceFrame(320), // 320采样 = 640字节 = 20ms@16kHz
			wantErr: false,
		},
		{
			name:    "大音量帧20ms无错误",
			frame:   loudFrame(320, 20000),
			wantErr: false,
		},
		{
			name:    "10ms帧(320字节)无错误",
			frame:   silenceFrame(160), // 160采样 = 320字节 = 10ms@16kHz
			wantErr: false,
		},
		{
			name:    "30ms帧(960字节)无错误",
			frame:   silenceFrame(480), // 480采样 = 960字节 = 30ms@16kHz
			wantErr: false,
		},
		{
			name:    "无效帧长度报错",
			frame:   silenceFrame(100), // 100采样 = 200字节，不是10/20/30ms
			wantErr: true,
		},
		{
			name:    "空帧报错",
			frame:   []byte{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := vad.IsSpeech(tt.frame)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// ---------- IsSpeech 静音 vs 有声检测 ----------

func TestVAD_IsSpeech_SilenceVsLoud(t *testing.T) {
	t.Parallel()

	vad, err := NewVAD(16000, VADVeryAggressive)
	require.NoError(t, err)

	// 静音帧应该不被检测为人声。
	silentResult, err := vad.IsSpeech(silenceFrame(320))
	require.NoError(t, err)
	assert.False(t, silentResult, "静音帧不应被检测为人声")
}

// ---------- ValidFrame 单元测试 ----------

func TestVAD_ValidFrame(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		sampleRate int
		samples    int // 采样数
		want       bool
	}{
		// 16kHz: 10ms=160, 20ms=320, 30ms=480 采样
		{name: "16kHz_10ms有效", sampleRate: 16000, samples: 160, want: true},
		{name: "16kHz_20ms有效", sampleRate: 16000, samples: 320, want: true},
		{name: "16kHz_30ms有效", sampleRate: 16000, samples: 480, want: true},
		{name: "16kHz_15ms无效", sampleRate: 16000, samples: 240, want: false},
		{name: "16kHz_0采样无效", sampleRate: 16000, samples: 0, want: false},
		{name: "16kHz_1采样无效", sampleRate: 16000, samples: 1, want: false},

		// 8kHz: 10ms=80, 20ms=160, 30ms=240 采样
		{name: "8kHz_10ms有效", sampleRate: 8000, samples: 80, want: true},
		{name: "8kHz_20ms有效", sampleRate: 8000, samples: 160, want: true},
		{name: "8kHz_30ms有效", sampleRate: 8000, samples: 240, want: true},
		{name: "8kHz_100采样无效", sampleRate: 8000, samples: 100, want: false},

		// 32kHz: 10ms=320, 20ms=640, 30ms=960 采样
		{name: "32kHz_10ms有效", sampleRate: 32000, samples: 320, want: true},
		{name: "32kHz_20ms有效", sampleRate: 32000, samples: 640, want: true},
		{name: "32kHz_30ms有效", sampleRate: 32000, samples: 960, want: true},
		{name: "32kHz_500采样无效", sampleRate: 32000, samples: 500, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			vad, err := NewVAD(tt.sampleRate, VADQuality)
			require.NoError(t, err)

			frame := silenceFrame(tt.samples)
			got := vad.ValidFrame(frame)
			assert.Equal(t, tt.want, got)
		})
	}
}

// ---------- VADMode 常量验证 ----------

func TestVADModeConstants(t *testing.T) {
	t.Parallel()

	assert.Equal(t, VADMode(0), VADQuality)
	assert.Equal(t, VADMode(1), VADLowBitrate)
	assert.Equal(t, VADMode(2), VADAggressive)
	assert.Equal(t, VADMode(3), VADVeryAggressive)
}

// ---------- 基准测试 ----------

func BenchmarkVAD_IsSpeech(b *testing.B) {
	vad, err := NewVAD(16000, VADVeryAggressive)
	require.NoError(b, err)

	frame := sineFrame(320) // 20ms@16kHz
	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		_, _ = vad.IsSpeech(frame)
	}
}

func BenchmarkVAD_ValidFrame(b *testing.B) {
	vad, err := NewVAD(16000, VADQuality)
	require.NoError(b, err)

	frame := silenceFrame(320)
	b.ResetTimer()

	for range b.N {
		vad.ValidFrame(frame)
	}
}
