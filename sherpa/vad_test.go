package sherpa

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- 配置测试 ---

func TestNewSileroVAD_EmptyModelPath(t *testing.T) {
	t.Parallel()

	_, err := NewSileroVAD(SileroVADConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ModelPath")
}

func TestSileroVADConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := SileroVADConfig{ModelPath: "/fake/model.onnx"}
	cfg.setDefaults()

	assert.InDelta(t, 0.5, float64(cfg.Threshold), 0.001)
	assert.InDelta(t, 0.5, float64(cfg.MinSilenceDuration), 0.001)
	assert.InDelta(t, 0.25, float64(cfg.MinSpeechDuration), 0.001)
	assert.InDelta(t, 10.0, float64(cfg.MaxSpeechDuration), 0.001)
	assert.Equal(t, 16000, cfg.SampleRate)
}

func TestSileroVADConfig_CustomValues(t *testing.T) {
	t.Parallel()

	cfg := SileroVADConfig{
		ModelPath:          "/model.onnx",
		Threshold:          0.4,
		MinSilenceDuration: 0.3,
		MinSpeechDuration:  0.1,
		MaxSpeechDuration:  5,
		SampleRate:         8000,
	}
	cfg.setDefaults()

	// 非零值不应被覆盖。
	assert.InDelta(t, 0.4, float64(cfg.Threshold), 0.001)
	assert.InDelta(t, 0.3, float64(cfg.MinSilenceDuration), 0.001)
	assert.InDelta(t, 0.1, float64(cfg.MinSpeechDuration), 0.001)
	assert.InDelta(t, 5.0, float64(cfg.MaxSpeechDuration), 0.001)
	assert.Equal(t, 8000, cfg.SampleRate)
}

// --- Close 幂等性 ---

func TestSileroVAD_Close_Idempotent(t *testing.T) {
	t.Parallel()

	// 直接构造一个已关闭的 VAD（不经过 NewSileroVAD，避免模型依赖）。
	v := &SileroVAD{engine: nil}

	// 重复 Close 不应 panic。
	assert.NoError(t, v.Close())
	assert.NoError(t, v.Close())
}

func TestSileroVAD_IsSpeech_AfterClose(t *testing.T) {
	t.Parallel()

	v := &SileroVAD{engine: nil}

	_, err := v.IsSpeech(make([]byte, 320))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

// --- IsSpeech 边界条件 ---

func TestSileroVAD_IsSpeech_EmptyFrame(t *testing.T) {
	t.Parallel()

	v := &SileroVAD{engine: nil}

	speech, err := v.IsSpeech(nil)
	assert.NoError(t, err)
	assert.False(t, speech)

	speech, err = v.IsSpeech([]byte{})
	assert.NoError(t, err)
	assert.False(t, speech)

	speech, err = v.IsSpeech([]byte{0x01})
	assert.NoError(t, err)
	assert.False(t, speech)
}

// --- SpeechSegments ---

func TestSileroVAD_SpeechSegments_AfterClose(t *testing.T) {
	t.Parallel()

	v := &SileroVAD{engine: nil}
	assert.Nil(t, v.SpeechSegments())
}

// --- Reset ---

func TestSileroVAD_Reset_AfterClose(t *testing.T) {
	t.Parallel()

	v := &SileroVAD{engine: nil, buf: []float32{1, 2, 3}}
	v.Reset()
	assert.Nil(t, v.buf)
}

// --- String ---

func TestSileroVAD_String(t *testing.T) {
	t.Parallel()

	v := &SileroVAD{sampleRate: 16000, windowSize: 512}
	s := v.String()
	assert.Contains(t, s, "16000")
	assert.Contains(t, s, "512")
}

// --- buildVADConfig ---

func TestBuildVADConfig(t *testing.T) {
	t.Parallel()

	cfg := SileroVADConfig{
		ModelPath:          "/model.onnx",
		Threshold:          0.4,
		MinSilenceDuration: 0.3,
		MinSpeechDuration:  0.1,
		MaxSpeechDuration:  8,
		SampleRate:         16000,
	}

	config := buildVADConfig(cfg)

	assert.Equal(t, "/model.onnx", config.SileroVad.Model)
	assert.InDelta(t, 0.4, float64(config.SileroVad.Threshold), 0.001)
	assert.InDelta(t, 0.3, float64(config.SileroVad.MinSilenceDuration), 0.001)
	assert.InDelta(t, 0.1, float64(config.SileroVad.MinSpeechDuration), 0.001)
	assert.InDelta(t, 8.0, float64(config.SileroVad.MaxSpeechDuration), 0.001)
	assert.Equal(t, sileroWindowSize, config.SileroVad.WindowSize)
	assert.Equal(t, 16000, config.SampleRate)
	assert.Equal(t, 1, config.NumThreads)
	assert.Equal(t, "cpu", config.Provider)
}

// --- Mock-based VAD tests ---

func TestSileroVAD_IsSpeech_WithMock_SpeechDetected(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{speechResult: true}
	v := &SileroVAD{
		engine:     mock,
		sampleRate: 16000,
		windowSize: 512,
	}

	// 生成 512 个采样（= 1 个窗口）的 PCM16 数据。
	pcmData := makePCM16Frame(512, 5000)
	speech, err := v.IsSpeech(pcmData)
	require.NoError(t, err)
	assert.True(t, speech)
	assert.Equal(t, 1, mock.acceptCount)
}

func TestSileroVAD_IsSpeech_WithMock_NoSpeech(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{speechResult: false}
	v := &SileroVAD{
		engine:     mock,
		sampleRate: 16000,
		windowSize: 512,
	}

	pcmData := makePCM16Frame(512, 100)
	speech, err := v.IsSpeech(pcmData)
	require.NoError(t, err)
	assert.False(t, speech)
}

func TestSileroVAD_IsSpeech_WithMock_BufferCarryOver(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{speechResult: true}
	v := &SileroVAD{
		engine:     mock,
		sampleRate: 16000,
		windowSize: 512,
	}

	// 发送 300 采样（不足一个窗口）。
	pcmData := makePCM16Frame(300, 5000)
	speech, err := v.IsSpeech(pcmData)
	require.NoError(t, err)
	assert.False(t, speech) // 不足一窗口，不处理
	assert.Equal(t, 0, mock.acceptCount)
	assert.Len(t, v.buf, 300) // 缓存了 300 采样

	// 再发送 300 采样（合计 600 >= 512，应处理一个窗口，余 88 缓存）。
	pcmData = makePCM16Frame(300, 5000)
	speech, err = v.IsSpeech(pcmData)
	require.NoError(t, err)
	assert.True(t, speech)
	assert.Equal(t, 1, mock.acceptCount)
	assert.Len(t, v.buf, 88) // 600 - 512 = 88
}

func TestSileroVAD_IsSpeech_WithMock_MultipleWindows(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{speechResult: false}
	v := &SileroVAD{
		engine:     mock,
		sampleRate: 16000,
		windowSize: 512,
	}

	// 发送 1200 采样（2 个完整窗口 + 176 余量）。
	pcmData := makePCM16Frame(1200, 5000)
	speech, err := v.IsSpeech(pcmData)
	require.NoError(t, err)
	assert.False(t, speech)
	assert.Equal(t, 2, mock.acceptCount) // 处理了 2 个窗口
	assert.Len(t, v.buf, 176)            // 1200 - 512*2 = 176
}

func TestSileroVAD_SpeechSegments_WithMock(t *testing.T) {
	t.Parallel()

	seg1 := []float32{0.1, 0.2, 0.3}
	seg2 := []float32{0.4, 0.5}
	mock := &fakeVADEngine{
		segments: [][]float32{seg1, seg2},
	}
	v := &SileroVAD{engine: mock}

	segments := v.SpeechSegments()
	require.Len(t, segments, 2)
	assert.Equal(t, seg1, segments[0])
	assert.Equal(t, seg2, segments[1])

	// 第二次应为空。
	segments = v.SpeechSegments()
	assert.Nil(t, segments)
}

func TestSileroVAD_SpeechSegments_WithMock_Empty(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{} // 无 segments
	v := &SileroVAD{engine: mock}

	segments := v.SpeechSegments()
	assert.Nil(t, segments)
}

func TestSileroVAD_Reset_WithMock(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{}
	v := &SileroVAD{
		engine: mock,
		buf:    []float32{1, 2, 3},
	}

	v.Reset()
	assert.True(t, mock.resetCalled)
	assert.Nil(t, v.buf)
}

func TestSileroVAD_Close_WithMock(t *testing.T) {
	t.Parallel()

	mock := &fakeVADEngine{}
	v := &SileroVAD{
		engine: mock,
		buf:    []float32{1, 2},
	}

	assert.NoError(t, v.Close())
	assert.True(t, mock.closed)
	assert.Nil(t, v.engine)
	assert.Nil(t, v.buf)
}

// --- 辅助函数 ---

// makePCM16Frame 生成指定采样数的 PCM16 LE 帧，值为交替正负。
func makePCM16Frame(samples int, amplitude int16) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		var val int16
		if i%2 == 0 {
			val = amplitude
		} else {
			val = -amplitude
		}
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(val))
	}
	return buf
}
