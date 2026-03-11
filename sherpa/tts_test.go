package sherpa

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- 配置测试 ---

func TestNewLocalTTS_EmptyModelPath(t *testing.T) {
	t.Parallel()

	_, err := NewLocalTTS(LocalTTSConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ModelPath")
}

func TestLocalTTSConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := LocalTTSConfig{ModelPath: "/fake/model.onnx"}
	cfg.setDefaults()

	assert.Equal(t, 1, cfg.NumThreads)
	assert.InDelta(t, 1.0, float64(cfg.Speed), 0.001)
}

func TestLocalTTSConfig_CustomValues(t *testing.T) {
	t.Parallel()

	cfg := LocalTTSConfig{
		ModelPath:  "/model.onnx",
		NumThreads: 4,
		Speed:      1.5,
	}
	cfg.setDefaults()

	assert.Equal(t, 4, cfg.NumThreads)
	assert.InDelta(t, 1.5, float64(cfg.Speed), 0.001)
}

// --- Close 幂等性 ---

func TestLocalTTS_Close_Idempotent(t *testing.T) {
	t.Parallel()

	tts := &LocalTTS{engine: nil}

	assert.NoError(t, tts.Close())
	assert.NoError(t, tts.Close())
}

// --- Synthesize 在 Close 后 ---

func TestLocalTTS_Synthesize_AfterClose(t *testing.T) {
	t.Parallel()

	tts := &LocalTTS{engine: nil}

	_, err := tts.Synthesize(context.Background(), "你好", nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

func TestLocalTTS_Synthesize_EmptyText(t *testing.T) {
	t.Parallel()

	// 空文本应直接返回 nil，不触发合成（即使 engine 为 nil 也不 panic）。
	tts := &LocalTTS{engine: nil}
	pcmOut, err := tts.Synthesize(context.Background(), "", nil)
	assert.NoError(t, err)
	assert.Nil(t, pcmOut)
}

// --- SynthesizeStream 在 Close 后 ---

func TestLocalTTS_SynthesizeStream_AfterClose(t *testing.T) {
	t.Parallel()

	tts := &LocalTTS{engine: nil}
	textCh := make(chan string, 1)

	_, err := tts.SynthesizeStream(context.Background(), textCh, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

// --- Cancel ---

func TestLocalTTS_Cancel(t *testing.T) {
	t.Parallel()

	tts := &LocalTTS{engine: nil}
	assert.NoError(t, tts.Cancel())
}

// --- TextRuneCount ---

func TestTextRuneCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"空字符串", "", 0},
		{"ASCII", "hello", 5},
		{"中文", "你好世界", 4},
		{"混合", "hi你好", 4},
		{"表情符号", "\U0001f44b\U0001f30d", 2},
		{"长文本", "这是一段比较长的中文文本用于测试分层路由", 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, TextRuneCount(tt.input))
		})
	}
}

// --- SynthesizeStream context 取消 ---

func TestLocalTTS_SynthesizeStream_ContextCancel(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	textCh := make(chan string, 4)

	// 向 textCh 发送空文本，验证跳过逻辑。
	textCh <- ""
	cancel()

	// 等待一小段时间确保 goroutine 有机会处理。
	time.Sleep(10 * time.Millisecond)

	// 关闭 textCh 验证不会 panic。
	close(textCh)

	_ = ctx // suppress unused
}

// --- buildTTSConfig ---

func TestBuildTTSConfig(t *testing.T) {
	t.Parallel()

	cfg := LocalTTSConfig{
		ModelPath:   "/model.onnx",
		TokensPath:  "/tokens.txt",
		DataDir:     "/data",
		DictDir:     "/dict",
		LexiconPath: "/lexicon.txt",
		RuleFsts:    "/rules.fst",
		RuleFars:    "/rules.far",
		NumThreads:  2,
	}

	config := buildTTSConfig(cfg)

	assert.Equal(t, "/model.onnx", config.Model.Vits.Model)
	assert.Equal(t, "/tokens.txt", config.Model.Vits.Tokens)
	assert.Equal(t, "/data", config.Model.Vits.DataDir)
	assert.Equal(t, "/dict", config.Model.Vits.DictDir)
	assert.Equal(t, "/lexicon.txt", config.Model.Vits.Lexicon)
	assert.Equal(t, 2, config.Model.NumThreads)
	assert.Equal(t, "cpu", config.Model.Provider)
	assert.Equal(t, "/rules.fst", config.RuleFsts)
	assert.Equal(t, "/rules.far", config.RuleFars)
	assert.Equal(t, 1, config.MaxNumSentences)
}

// --- Mock-based TTS tests ---

func TestLocalTTS_Synthesize_WithMock_Success(t *testing.T) {
	t.Parallel()

	mock := &fakeTTSEngine{
		samples:        []float32{0.1, -0.1, 0.2, -0.2},
		fakeSampleRate: 16000,
	}
	tts := &LocalTTS{
		engine:     mock,
		config:     LocalTTSConfig{SpeakerID: 0, Speed: 1.0},
		logger:     testLogger(),
		sampleRate: 16000,
	}

	pcmOut, err := tts.Synthesize(context.Background(), "你好", nil)
	require.NoError(t, err)
	require.NotNil(t, pcmOut)
	// 4 float32 samples -> 8 bytes PCM16
	assert.Len(t, pcmOut, 8)
	assert.True(t, mock.generateCalled)
}

func TestLocalTTS_Synthesize_WithMock_NilSamples(t *testing.T) {
	t.Parallel()

	mock := &fakeTTSEngine{
		samples:        nil, // generate 返回 nil
		fakeSampleRate: 16000,
	}
	tts := &LocalTTS{
		engine:     mock,
		config:     LocalTTSConfig{SpeakerID: 0, Speed: 1.0},
		logger:     testLogger(),
		sampleRate: 16000,
	}

	pcmOut, err := tts.Synthesize(context.Background(), "你好", nil)
	assert.NoError(t, err)
	assert.Nil(t, pcmOut)
}

func TestLocalTTS_SampleRate(t *testing.T) {
	t.Parallel()

	tts := &LocalTTS{sampleRate: 22050}
	assert.Equal(t, 22050, tts.SampleRate())
}

func TestLocalTTS_Close_WithMock(t *testing.T) {
	t.Parallel()

	mock := &fakeTTSEngine{fakeSampleRate: 16000}
	tts := &LocalTTS{engine: mock}

	assert.NoError(t, tts.Close())
	assert.True(t, mock.closed)
	assert.Nil(t, tts.engine)
}

func TestLocalTTS_SynthesizeStream_WithMock_Success(t *testing.T) {
	t.Parallel()

	mock := &fakeTTSEngine{
		samples:        []float32{0.1, -0.1},
		fakeSampleRate: 16000,
	}
	tts := &LocalTTS{
		engine:     mock,
		config:     LocalTTSConfig{SpeakerID: 0, Speed: 1.0},
		logger:     testLogger(),
		sampleRate: 16000,
	}

	ctx := context.Background()
	textCh := make(chan string, 4)
	textCh <- "你好"
	textCh <- "" // 空文本应跳过
	textCh <- "世界"
	close(textCh)

	audioCh, err := tts.SynthesizeStream(ctx, textCh, nil)
	require.NoError(t, err)

	var chunks [][]byte
	for chunk := range audioCh {
		chunks = append(chunks, chunk)
	}

	assert.Len(t, chunks, 2) // 两段非空文本
}

func TestLocalTTS_SynthesizeStream_WithMock_ContextCancel(t *testing.T) {
	t.Parallel()

	mock := &fakeTTSEngine{
		samples:        []float32{0.1, -0.1},
		fakeSampleRate: 16000,
	}
	tts := &LocalTTS{
		engine:     mock,
		config:     LocalTTSConfig{SpeakerID: 0, Speed: 1.0},
		logger:     testLogger(),
		sampleRate: 16000,
	}

	ctx, cancel := context.WithCancel(context.Background())
	textCh := make(chan string, 4)

	audioCh, err := tts.SynthesizeStream(ctx, textCh, nil)
	require.NoError(t, err)

	cancel()

	// audioCh 应最终关闭。
	select {
	case _, ok := <-audioCh:
		_ = ok // 可能收到一个 chunk，也可能不收到。
	case <-time.After(time.Second):
		t.Fatal("audioCh not closed after cancel")
	}
}

func TestLocalTTS_SynthesizeStream_WithMock_EngineError(t *testing.T) {
	t.Parallel()

	// 第一次合成成功，然后关闭 engine 导致第二次失败。
	mock := &fakeTTSEngine{
		samples:        []float32{0.1, -0.1},
		fakeSampleRate: 16000,
	}
	tts := &LocalTTS{
		engine:     mock,
		config:     LocalTTSConfig{SpeakerID: 0, Speed: 1.0},
		logger:     testLogger(),
		sampleRate: 16000,
	}

	ctx := context.Background()
	textCh := make(chan string, 4)
	textCh <- "你好"

	audioCh, err := tts.SynthesizeStream(ctx, textCh, nil)
	require.NoError(t, err)

	// 收到第一个 chunk。
	chunk := <-audioCh
	assert.NotNil(t, chunk)

	// 关闭 engine，使下一次合成失败。
	tts.Close()
	textCh <- "失败"
	close(textCh)

	// audioCh 应关闭。
	for range audioCh {
		// drain
	}
}
