package sherpa

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/omeyang/Sonata/engine/aiface"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- 配置测试 ---

func TestNewLocalASR_EmptyTokensPath(t *testing.T) {
	t.Parallel()

	_, err := NewLocalASR(LocalASRConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "TokensPath")
}

func TestLocalASRConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := LocalASRConfig{TokensPath: "/fake/tokens.txt"}
	cfg.setDefaults()

	assert.Equal(t, 1, cfg.NumThreads)
	assert.Equal(t, 16000, cfg.SampleRate)
	assert.InDelta(t, 2.4, float64(cfg.Rule1MinTrailingSilence), 0.001)
	assert.InDelta(t, 1.2, float64(cfg.Rule2MinTrailingSilence), 0.001)
	assert.InDelta(t, 20.0, float64(cfg.Rule3MinUtteranceLength), 0.001)
}

func TestLocalASRConfig_CustomValues(t *testing.T) {
	t.Parallel()

	cfg := LocalASRConfig{
		TokensPath:              "/tokens.txt",
		NumThreads:              4,
		SampleRate:              8000,
		Rule1MinTrailingSilence: 1.0,
		Rule2MinTrailingSilence: 0.5,
		Rule3MinUtteranceLength: 10,
	}
	cfg.setDefaults()

	// 非零值不应被覆盖。
	assert.Equal(t, 4, cfg.NumThreads)
	assert.Equal(t, 8000, cfg.SampleRate)
	assert.InDelta(t, 1.0, float64(cfg.Rule1MinTrailingSilence), 0.001)
	assert.InDelta(t, 0.5, float64(cfg.Rule2MinTrailingSilence), 0.001)
	assert.InDelta(t, 10.0, float64(cfg.Rule3MinUtteranceLength), 0.001)
}

// --- Close 幂等性 ---

func TestLocalASR_Close_Idempotent(t *testing.T) {
	t.Parallel()

	a := &LocalASR{engine: nil}

	// 重复 Close 不应 panic。
	assert.NoError(t, a.Close())
	assert.NoError(t, a.Close())
}

// --- StartStream 在 Close 后 ---

func TestLocalASR_StartStream_AfterClose(t *testing.T) {
	t.Parallel()

	a := &LocalASR{engine: nil}

	_, err := a.StartStream(context.Background(), aiface.ASRConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

// --- localASRStream ---

func TestLocalASRStream_FeedAudio_AfterClose(t *testing.T) {
	t.Parallel()

	s := &localASRStream{
		closed:   true,
		events:   make(chan aiface.ASREvent, 1),
		parentMu: &sync.Mutex{},
	}

	err := s.FeedAudio(context.Background(), make([]byte, 320))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

func TestLocalASRStream_FeedAudio_EmptyChunk(t *testing.T) {
	t.Parallel()

	s := &localASRStream{
		events:   make(chan aiface.ASREvent, 1),
		parentMu: &sync.Mutex{},
	}

	// nil chunk 应无操作。
	assert.NoError(t, s.FeedAudio(context.Background(), nil))

	// 空 chunk 应无操作。
	assert.NoError(t, s.FeedAudio(context.Background(), []byte{}))

	// 单字节（不足一个采样）应无操作。
	assert.NoError(t, s.FeedAudio(context.Background(), []byte{0x01}))
}

func TestLocalASRStream_Events_ReturnsChannel(t *testing.T) {
	t.Parallel()

	ch := make(chan aiface.ASREvent, 8)
	s := &localASRStream{
		events: ch,
	}

	assert.Equal(t, (<-chan aiface.ASREvent)(ch), s.Events())
}

func TestLocalASRStream_Close_Idempotent(t *testing.T) {
	t.Parallel()

	_, cancel := context.WithCancel(context.Background())
	s := &localASRStream{
		events:   make(chan aiface.ASREvent, 1),
		cancel:   cancel,
		parentMu: &sync.Mutex{},
	}

	assert.NoError(t, s.Close())
	// 第二次 Close 不应 panic。
	assert.NoError(t, s.Close())
}

// --- decodeInterval ---

func TestDecodeInterval(t *testing.T) {
	t.Parallel()

	// 验证解码间隔在合理范围内（50ms-500ms）。
	assert.GreaterOrEqual(t, decodeInterval.Milliseconds(), int64(50))
	assert.LessOrEqual(t, decodeInterval.Milliseconds(), int64(500))
}

// --- buildASRConfig ---

func TestBuildASRConfig(t *testing.T) {
	t.Parallel()

	cfg := LocalASRConfig{
		EncoderPath:             "/encoder.onnx",
		DecoderPath:             "/decoder.onnx",
		TokensPath:              "/tokens.txt",
		ModelType:               "paraformer",
		NumThreads:              2,
		SampleRate:              16000,
		EnableEndpoint:          true,
		Rule1MinTrailingSilence: 2.0,
		Rule2MinTrailingSilence: 1.0,
		Rule3MinUtteranceLength: 15,
	}

	config := buildASRConfig(cfg)

	assert.Equal(t, "/encoder.onnx", config.ModelConfig.Paraformer.Encoder)
	assert.Equal(t, "/decoder.onnx", config.ModelConfig.Paraformer.Decoder)
	assert.Equal(t, "/tokens.txt", config.ModelConfig.Tokens)
	assert.Equal(t, "paraformer", config.ModelConfig.ModelType)
	assert.Equal(t, 2, config.ModelConfig.NumThreads)
	assert.Equal(t, "cpu", config.ModelConfig.Provider)
	assert.Equal(t, "greedy_search", config.DecodingMethod)
	assert.Equal(t, 1, config.EnableEndpoint)
	assert.Equal(t, 16000, config.FeatConfig.SampleRate)
	assert.Equal(t, 80, config.FeatConfig.FeatureDim)
	assert.InDelta(t, 2.0, float64(config.Rule1MinTrailingSilence), 0.001)
	assert.InDelta(t, 1.0, float64(config.Rule2MinTrailingSilence), 0.001)
	assert.InDelta(t, 15.0, float64(config.Rule3MinUtteranceLength), 0.001)
}

func TestBuildASRConfig_EndpointDisabled(t *testing.T) {
	t.Parallel()

	cfg := LocalASRConfig{
		TokensPath:     "/tokens.txt",
		EnableEndpoint: false,
	}
	cfg.setDefaults()

	config := buildASRConfig(cfg)
	assert.Equal(t, 0, config.EnableEndpoint)
}

// --- Mock-based ASR tests ---

func TestLocalASR_StartStream_Success(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{}
	mockEngine := &fakeASREngine{stream: mockStream}

	a := &LocalASR{
		engine: mockEngine,
		config: LocalASRConfig{SampleRate: 16000},
		logger: testLogger(),
	}

	stream, err := a.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	require.NotNil(t, stream)

	// 关闭流。
	assert.NoError(t, stream.Close())
	// 等待 decodeLoop goroutine 退出。
	time.Sleep(20 * time.Millisecond)
}

func TestLocalASR_StartStream_CreateStreamFails(t *testing.T) {
	t.Parallel()

	mockEngine := &fakeASREngine{err: assert.AnError}

	a := &LocalASR{
		engine: mockEngine,
		config: LocalASRConfig{SampleRate: 16000},
		logger: testLogger(),
	}

	_, err := a.StartStream(context.Background(), aiface.ASRConfig{})
	require.Error(t, err)
}

func TestLocalASR_Close_WithEngine(t *testing.T) {
	t.Parallel()

	mockEngine := &fakeASREngine{}
	a := &LocalASR{engine: mockEngine}

	assert.NoError(t, a.Close())
	assert.True(t, mockEngine.closed)
	assert.Nil(t, a.engine)
}

func TestLocalASRStream_FeedAudio_WithMock(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{}
	mu := &sync.Mutex{}

	s := &localASRStream{
		stream:     mockStream,
		events:     make(chan aiface.ASREvent, 1),
		sampleRate: 16000,
		parentMu:   mu,
	}

	// 发送有效音频。
	pcmData := makePCM16Frame(160, 1000)
	err := s.FeedAudio(context.Background(), pcmData)
	assert.NoError(t, err)
	assert.True(t, mockStream.waveformAccepted)
}

func TestLocalASRStream_DecodeTick_PartialText(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 1,
		resultText: "你好",
		endpoint:   false,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mu := &sync.Mutex{}
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent, 32),
		parentMu: mu,
		startAt:  time.Now(),
	}

	s.decodeTick(ctx)

	select {
	case evt := <-s.events:
		assert.False(t, evt.IsFinal)
		assert.Equal(t, "你好", evt.Text)
	default:
		t.Fatal("expected partial event")
	}
}

func TestLocalASRStream_DecodeTick_FinalText(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 1,
		resultText: "你好世界",
		endpoint:   true,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mu := &sync.Mutex{}
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent, 32),
		parentMu: mu,
		startAt:  time.Now(),
	}

	s.decodeTick(ctx)

	select {
	case evt := <-s.events:
		assert.True(t, evt.IsFinal)
		assert.Equal(t, "你好世界", evt.Text)
		assert.InDelta(t, 1.0, evt.Confidence, 0.001)
	default:
		t.Fatal("expected final event")
	}

	// 验证 reset 被调用。
	assert.True(t, mockStream.resetCalled)
}

func TestLocalASRStream_DecodeTick_EmptyEndpoint(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 0,
		resultText: "",
		endpoint:   true,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mu := &sync.Mutex{}
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent, 32),
		parentMu: mu,
		startAt:  time.Now(),
	}

	s.decodeTick(ctx)

	// 空文本 endpoint 不应发送事件。
	select {
	case <-s.events:
		t.Fatal("should not send event for empty endpoint")
	default:
		// OK
	}

	// 但应调用 reset。
	assert.True(t, mockStream.resetCalled)
}

func TestLocalASRStream_DecodeTick_DuplicatePartialSkipped(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 1,
		resultText: "你好",
		endpoint:   false,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mu := &sync.Mutex{}
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent, 32),
		parentMu: mu,
		lastText: "你好", // 已发送过相同文本
		startAt:  time.Now(),
	}

	s.decodeTick(ctx)

	// 相同文本不应重复发送。
	select {
	case <-s.events:
		t.Fatal("should not send duplicate partial")
	default:
		// OK
	}
}

func TestLocalASRStream_DecodeTick_ContextCancel(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 1,
		resultText: "你好",
		endpoint:   true,
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // 提前取消

	mu := &sync.Mutex{}
	// events 通道容量为 0，且 context 已取消，final 发送应通过 ctx.Done 退出。
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent), // 无缓冲
		parentMu: mu,
		startAt:  time.Now(),
	}

	// 不应阻塞。
	s.decodeTick(ctx)
}

func TestLocalASRStream_DecodeLoop_ExitsOnCancel(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{}
	ctx, cancel := context.WithCancel(context.Background())

	mu := &sync.Mutex{}
	s := &localASRStream{
		stream:   mockStream,
		events:   make(chan aiface.ASREvent, 32),
		parentMu: mu,
		startAt:  time.Now(),
	}

	done := make(chan struct{})
	go func() {
		s.decodeLoop(ctx)
		close(done)
	}()

	cancel()

	select {
	case <-done:
		// OK - goroutine exited
	case <-time.After(time.Second):
		t.Fatal("decodeLoop did not exit after cancel")
	}

	// events 通道应关闭。
	_, ok := <-s.events
	assert.False(t, ok)
}

func TestLocalASRStream_DecodeTick_PartialDroppedWhenFull(t *testing.T) {
	t.Parallel()

	mockStream := &fakeASRStream{
		readyCount: 1,
		resultText: "你好",
		endpoint:   false,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mu := &sync.Mutex{}
	// 容量为 1 的 events 通道，先填满。
	events := make(chan aiface.ASREvent, 1)
	events <- aiface.ASREvent{Text: "占位"}

	s := &localASRStream{
		stream:   mockStream,
		events:   events,
		parentMu: mu,
		startAt:  time.Now(),
	}

	// 不应阻塞（partial 通道满时丢弃）。
	s.decodeTick(ctx)

	// 通道中应只有原来的占位事件。
	evt := <-events
	assert.Equal(t, "占位", evt.Text)
}
