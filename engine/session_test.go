package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/omeyang/Sonata/engine/aiface"
	"github.com/omeyang/Sonata/engine/mediafsm"
)

// ---------------------------------------------------------------------------
// Mock 实现
// ---------------------------------------------------------------------------

// mockTransport 模拟音频传输层。
type mockTransport struct {
	audioCh      chan []byte
	playedAudio  [][]byte
	mu           sync.Mutex
	playErr      error
	stopErr      error
	closeErr     error
	playBlocking bool // 如果为 true，PlayAudio 会阻塞直到 ctx 取消。
}

func newMockTransport() *mockTransport {
	return &mockTransport{
		audioCh: make(chan []byte, 100),
	}
}

func (m *mockTransport) AudioIn() <-chan []byte {
	return m.audioCh
}

func (m *mockTransport) PlayAudio(ctx context.Context, audio []byte) error {
	if m.playBlocking {
		<-ctx.Done()
		return ctx.Err()
	}
	m.mu.Lock()
	m.playedAudio = append(m.playedAudio, audio)
	m.mu.Unlock()
	if m.playErr != nil {
		return m.playErr
	}
	return nil
}

func (m *mockTransport) StopPlayback(_ context.Context) error {
	return m.stopErr
}

func (m *mockTransport) Close() error {
	return m.closeErr
}

func (m *mockTransport) getPlayedAudio() [][]byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([][]byte, len(m.playedAudio))
	copy(result, m.playedAudio)
	return result
}

// mockDialogue 模拟对话引擎。
type mockDialogue struct {
	opening      string
	processReply string
	processErr   error
	streamCh     chan string
	streamErr    error
	finished     bool
	mu           sync.Mutex
}

func (m *mockDialogue) Opening() string {
	return m.opening
}

func (m *mockDialogue) Process(_ context.Context, _ string) (string, error) {
	return m.processReply, m.processErr
}

func (m *mockDialogue) ProcessStream(_ context.Context, _ string) (<-chan string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	return m.streamCh, nil
}

func (m *mockDialogue) Finished() bool {
	return m.finished
}

// setStreamCh 线程安全地设置流式回复通道。
func (m *mockDialogue) setStreamCh(ch chan string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.streamCh = ch
}

// mockASRProvider 模拟 ASR 提供者。
type mockASRProvider struct {
	stream    *mockASRStream
	streamErr error
}

func (m *mockASRProvider) StartStream(_ context.Context, _ aiface.ASRConfig) (aiface.ASRStream, error) {
	if m.streamErr != nil {
		return nil, m.streamErr
	}
	return m.stream, nil
}

// mockASRStream 模拟 ASR 流。
type mockASRStream struct {
	eventsCh chan aiface.ASREvent
	feedErr  error
	closeErr error
	fedData  [][]byte
	mu       sync.Mutex
}

func newMockASRStream() *mockASRStream {
	return &mockASRStream{
		eventsCh: make(chan aiface.ASREvent, 16),
	}
}

func (m *mockASRStream) FeedAudio(_ context.Context, chunk []byte) error {
	m.mu.Lock()
	m.fedData = append(m.fedData, chunk)
	m.mu.Unlock()
	return m.feedErr
}

func (m *mockASRStream) Events() <-chan aiface.ASREvent {
	return m.eventsCh
}

func (m *mockASRStream) Close() error {
	return m.closeErr
}

func (m *mockASRStream) getFedData() [][]byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([][]byte, len(m.fedData))
	copy(result, m.fedData)
	return result
}

// mockTTSProvider 模拟 TTS 提供者。
type mockTTSProvider struct {
	synthData []byte
	synthErr  error
	cancelErr error
	cancelled atomic.Bool
}

func (m *mockTTSProvider) SynthesizeStream(_ context.Context, textCh <-chan string, _ aiface.TTSConfig) (<-chan []byte, error) {
	ch := make(chan []byte, 16)
	go func() {
		defer close(ch)
		for range textCh {
			if len(m.synthData) > 0 {
				ch <- m.synthData
			}
		}
	}()
	return ch, nil
}

func (m *mockTTSProvider) Synthesize(_ context.Context, _ string, _ aiface.TTSConfig) ([]byte, error) {
	if m.synthErr != nil {
		return nil, m.synthErr
	}
	return m.synthData, nil
}

func (m *mockTTSProvider) Cancel() error {
	m.cancelled.Store(true)
	return m.cancelErr
}

// mockSpeechDetector 模拟人声检测器。
type mockSpeechDetector struct {
	isSpeech bool
	err      error
}

func (m *mockSpeechDetector) IsSpeech(_ []byte) (bool, error) {
	return m.isSpeech, m.err
}

// mockWarmer 模拟实现 Warmer 接口的提供者。
type mockWarmer struct {
	warmupErr error
	called    atomic.Bool
}

func (m *mockWarmer) Warmup(_ context.Context) error {
	m.called.Store(true)
	return m.warmupErr
}

// mockASRProviderWithWarmer 同时实现 ASRProvider 和 Warmer。
type mockASRProviderWithWarmer struct {
	mockASRProvider
	mockWarmer
}

func (m *mockASRProviderWithWarmer) Warmup(ctx context.Context) error {
	return m.mockWarmer.Warmup(ctx)
}

// mockMetrics 记录调用的 Metrics 实现。
type mockMetrics struct {
	mu              sync.Mutex
	asrLatencies    []time.Duration
	llmFirstTokens  []time.Duration
	ttsFirstChunks  []time.Duration
	turnLatencies   []time.Duration
	bargeInCount    int
	silenceTimeouts int
	providerErrors  []string
	callActive      int
	fillerPlayed    int
	speculativeHits int
	speculativeMiss int
}

func (m *mockMetrics) RecordASRLatency(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.asrLatencies = append(m.asrLatencies, d)
}

func (m *mockMetrics) RecordLLMFirstToken(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.llmFirstTokens = append(m.llmFirstTokens, d)
}

func (m *mockMetrics) RecordTTSFirstChunk(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ttsFirstChunks = append(m.ttsFirstChunks, d)
}

func (m *mockMetrics) RecordTurnLatency(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.turnLatencies = append(m.turnLatencies, d)
}

func (m *mockMetrics) IncBargeIn() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bargeInCount++
}

func (m *mockMetrics) IncSilenceTimeout() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.silenceTimeouts++
}

func (m *mockMetrics) IncProviderError(provider string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.providerErrors = append(m.providerErrors, provider)
}

func (m *mockMetrics) SetCallActive(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callActive = count
}

func (m *mockMetrics) IncFillerPlayed() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fillerPlayed++
}

func (m *mockMetrics) IncSpeculativeHit() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.speculativeHits++
}

func (m *mockMetrics) IncSpeculativeMiss() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.speculativeMiss++
}

// ---------------------------------------------------------------------------
// 辅助函数
// ---------------------------------------------------------------------------

// silenceFrame160 生成 20ms 8kHz 静音帧（全零 PCM16 LE，160 采样）。
func silenceFrame160() []byte {
	return make([]byte, 160*2)
}

// loudFrame 生成大音量帧（所有采样设为 amplitude）。
func loudFrame(samples int, amplitude int16) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(amplitude))
	}
	return buf
}

// appTransitions 返回简化的 APP 场景转换表（从 Idle 直接开始对话）。
func appTransitions() []mediafsm.Transition {
	return mediafsm.AppTransitions()
}

// minimalConfig 构造最小可运行的配置。
func minimalConfig(t *mockTransport) Config {
	return Config{
		SessionID:   "test-001",
		Transport:   t,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         5,
			MaxSilenceSec:          2,
			FirstSilenceTimeoutSec: 1,
		},
	}
}

// ---------------------------------------------------------------------------
// 测试：New()
// ---------------------------------------------------------------------------

// TestNewDefaults 验证 New() 正确填充默认值。
func TestNewDefaults(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:   "s-1",
		Transport:   trans,
		Transitions: appTransitions(),
	})

	require.NotNil(t, s)
	assert.Equal(t, "s-1", s.cfg.SessionID)
	assert.Equal(t, 8000, s.cfg.InputSampleRate, "默认输入采样率应为8000")
	assert.Equal(t, 16000, s.cfg.ASRSampleRate, "默认ASR采样率应为16000")
	assert.Equal(t, -35.0, s.cfg.EnergyThresholdDBFS, "默认能量阈值应为-35.0")
	assert.Equal(t, 300, s.cfg.Protection.MaxDurationSec, "默认最大时长应为300秒")
	assert.Equal(t, 15, s.cfg.Protection.MaxSilenceSec, "默认最大静默应为15秒")
	assert.Equal(t, 6, s.cfg.Protection.FirstSilenceTimeoutSec, "默认首次静默超时应为6秒")
	assert.NotNil(t, s.mfsm)
	assert.NotNil(t, s.logger)
	assert.NotNil(t, s.asrResults)
	assert.NotNil(t, s.botDoneCh)
}

// TestNewCustomValues 验证 New() 保留自定义配置值。
func TestNewCustomValues(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	detector := &mockSpeechDetector{isSpeech: true}
	s := New(Config{
		SessionID:           "s-2",
		Transport:           trans,
		Transitions:         appTransitions(),
		InputSampleRate:     16000,
		ASRSampleRate:       16000,
		EnergyThresholdDBFS: -40.0,
		SpeechDetector:      detector,
		Protection: ProtectionConfig{
			MaxDurationSec:         600,
			MaxSilenceSec:          30,
			FirstSilenceTimeoutSec: 10,
		},
	})

	assert.Equal(t, 16000, s.cfg.InputSampleRate)
	assert.Equal(t, 16000, s.cfg.ASRSampleRate)
	assert.Equal(t, -40.0, s.cfg.EnergyThresholdDBFS)
	assert.Equal(t, 600, s.cfg.Protection.MaxDurationSec)
	assert.Equal(t, 30, s.cfg.Protection.MaxSilenceSec)
	assert.Equal(t, 10, s.cfg.Protection.FirstSilenceTimeoutSec)
	assert.Equal(t, detector, s.speechDetector)
}

// ---------------------------------------------------------------------------
// 测试：FSM()
// ---------------------------------------------------------------------------

// TestFSM 验证 FSM() 返回有效的状态机。
func TestFSM(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(minimalConfig(trans))

	fsm := s.FSM()
	require.NotNil(t, fsm)
	assert.Equal(t, mediafsm.Idle, fsm.State())
}

// ---------------------------------------------------------------------------
// 测试：RecordEvent 线程安全
// ---------------------------------------------------------------------------

// TestRecordEventThreadSafety 验证 RecordEvent 在并发调用下不会 panic 或丢失数据。
func TestRecordEventThreadSafety(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()

	const goroutines = 50
	const eventsPerGoroutine = 20

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for range goroutines {
		go func() {
			defer wg.Done()
			for range eventsPerGoroutine {
				s.RecordEvent(EventHangup, map[string]string{"key": "value"})
			}
		}()
	}

	wg.Wait()

	s.mu.Lock()
	assert.Equal(t, goroutines*eventsPerGoroutine, len(s.events))
	s.mu.Unlock()
}

// ---------------------------------------------------------------------------
// 测试：buildResult
// ---------------------------------------------------------------------------

// TestBuildResult 验证 buildResult 正确构造结果。
func TestBuildResult(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:   "result-test",
		Transport:   trans,
		Transitions: appTransitions(),
	})
	s.startTime = time.Now().Add(-10 * time.Second)

	s.RecordEvent(EventBotSpeakStart, map[string]string{"text": "hello"})
	s.RecordEvent(EventUserSpeechStart, nil)
	s.RecordEvent(EventHangup, map[string]string{"cause": "normal"})

	result := s.buildResult()

	require.NotNil(t, result)
	assert.Equal(t, "result-test", result.SessionID)
	assert.GreaterOrEqual(t, result.Duration, 10, "时长应大于等于10秒")
	assert.Len(t, result.Events, 3)
	assert.Equal(t, EventBotSpeakStart, result.Events[0].EventType)
	assert.Equal(t, "hello", result.Events[0].Metadata["text"])
}

// TestBuildResultEmptyEvents 验证无事件时返回空切片。
func TestBuildResultEmptyEvents(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()

	result := s.buildResult()
	assert.Empty(t, result.Events)
	assert.Equal(t, "test-001", result.SessionID)
}

// ---------------------------------------------------------------------------
// 测试：isSpeechFrame
// ---------------------------------------------------------------------------

// TestIsSpeechFrame 验证语音检测逻辑。
func TestIsSpeechFrame(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		detector SpeechDetector
		frame    []byte
		want     bool
	}{
		{
			name:     "SpeechDetector检测到人声",
			detector: &mockSpeechDetector{isSpeech: true},
			frame:    silenceFrame160(),
			want:     true,
		},
		{
			name:     "SpeechDetector未检测到人声",
			detector: &mockSpeechDetector{isSpeech: false},
			frame:    loudFrame(160, 10000),
			want:     false,
		},
		{
			name:     "SpeechDetector出错_退回能量检测_大音量",
			detector: &mockSpeechDetector{err: errors.New("vad error")},
			frame:    loudFrame(160, 20000),
			want:     true, // 大音量帧能量高于 -35 dBFS 阈值。
		},
		{
			name:     "SpeechDetector出错_退回能量检测_静音",
			detector: &mockSpeechDetector{err: errors.New("vad error")},
			frame:    silenceFrame160(),
			want:     false, // 静音帧能量为 -96 dBFS，低于阈值。
		},
		{
			name:     "无SpeechDetector_大音量",
			detector: nil,
			frame:    loudFrame(160, 20000),
			want:     true,
		},
		{
			name:     "无SpeechDetector_静音",
			detector: nil,
			frame:    silenceFrame160(),
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			trans := newMockTransport()
			s := New(Config{
				SessionID:      "test",
				Transport:      trans,
				Transitions:    appTransitions(),
				SpeechDetector: tt.detector,
			})
			assert.Equal(t, tt.want, s.isSpeechFrame(tt.frame))
		})
	}
}

// ---------------------------------------------------------------------------
// 测试：feedASR
// ---------------------------------------------------------------------------

// TestFeedASR 验证音频帧送入 ASR 流的逻辑。
func TestFeedASR(t *testing.T) {
	t.Parallel()

	t.Run("nil流不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		assert.NotPanics(t, func() {
			s.feedASR(context.Background(), silenceFrame160())
		})
	})

	t.Run("8kHz到16kHz重采样", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		trans := newMockTransport()
		s := New(Config{
			SessionID:       "test",
			Transport:       trans,
			Transitions:     appTransitions(),
			InputSampleRate: 8000,
			ASRSampleRate:   16000,
		})
		s.asrStream = stream

		// 8kHz 20ms = 160 采样 = 320 字节。
		frame := loudFrame(160, 1000)
		s.feedASR(context.Background(), frame)

		fed := stream.getFedData()
		require.Len(t, fed, 1)
		// 重采样后应为 2 倍大小（320 → 640 字节）。
		assert.Equal(t, len(frame)*2, len(fed[0]), "重采样后数据应为2倍")
	})

	t.Run("相同采样率不重采样", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		trans := newMockTransport()
		s := New(Config{
			SessionID:       "test",
			Transport:       trans,
			Transitions:     appTransitions(),
			InputSampleRate: 16000,
			ASRSampleRate:   16000,
		})
		s.asrStream = stream

		frame := loudFrame(320, 1000)
		s.feedASR(context.Background(), frame)

		fed := stream.getFedData()
		require.Len(t, fed, 1)
		assert.Equal(t, frame, fed[0], "相同采样率不应重采样")
	})

	t.Run("FeedAudio返回错误不panic", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		stream.feedErr = errors.New("feed failed")
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.asrStream = stream

		assert.NotPanics(t, func() {
			s.feedASR(context.Background(), loudFrame(160, 1000))
		})
	})

	t.Run("空帧重采样后为nil不送入", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		trans := newMockTransport()
		s := New(Config{
			SessionID:       "test",
			Transport:       trans,
			Transitions:     appTransitions(),
			InputSampleRate: 8000,
			ASRSampleRate:   16000,
		})
		s.asrStream = stream

		// 空帧或过短帧，Resample8to16 返回 nil。
		s.feedASR(context.Background(), []byte{})

		fed := stream.getFedData()
		assert.Empty(t, fed, "空帧不应送入 ASR")
	})
}

// ---------------------------------------------------------------------------
// 测试：warmupProviders
// ---------------------------------------------------------------------------

// TestWarmupProviders 验证预热逻辑。
func TestWarmupProviders(t *testing.T) {
	t.Parallel()

	t.Run("实现Warmer的提供者被调用", func(t *testing.T) {
		t.Parallel()
		warmerASR := &mockASRProviderWithWarmer{
			mockASRProvider: mockASRProvider{stream: newMockASRStream()},
		}
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         warmerASR,
		})

		s.warmupProviders(context.Background())
		assert.True(t, warmerASR.called.Load(), "Warmer.Warmup 应被调用")
	})

	t.Run("不实现Warmer的提供者被跳过", func(t *testing.T) {
		t.Parallel()
		asr := &mockASRProvider{stream: newMockASRStream()}
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         asr,
		})

		// 不应 panic。
		assert.NotPanics(t, func() {
			s.warmupProviders(context.Background())
		})
	})

	t.Run("Warmup返回错误不中断", func(t *testing.T) {
		t.Parallel()
		warmerASR := &mockASRProviderWithWarmer{
			mockASRProvider: mockASRProvider{stream: newMockASRStream()},
			mockWarmer:      mockWarmer{warmupErr: errors.New("warmup failed")},
		}
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         warmerASR,
		})

		assert.NotPanics(t, func() {
			s.warmupProviders(context.Background())
		})
		assert.True(t, warmerASR.called.Load())
	})

	t.Run("nil提供者不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
		})

		assert.NotPanics(t, func() {
			s.warmupProviders(context.Background())
		})
	})
}

// ---------------------------------------------------------------------------
// 测试：Run() 基本流程
// ---------------------------------------------------------------------------

// TestRunAudioClosed 验证音频通道关闭时 Run 正常退出。
func TestRunAudioClosed(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(minimalConfig(trans))

	// 立即关闭音频通道，Run 应退出。
	close(trans.audioCh)

	result, err := s.Run(context.Background())

	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "test-001", result.SessionID)
}

// TestRunContextCancellation 验证 context 取消时 Run 正确退出。
func TestRunContextCancellation(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:   "cancel-test",
		Transport:   trans,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         1, // 1秒后超时。
			MaxSilenceSec:          10,
			FirstSilenceTimeoutSec: 10,
		},
	})

	result, err := s.Run(context.Background())

	require.Error(t, err, "超时后应返回错误")
	assert.Contains(t, err.Error(), "context")
	require.NotNil(t, result)
	assert.Equal(t, "cancel-test", result.SessionID)
}

// TestRunWithDialogue 验证有对话引擎时的开场白流程。
func TestRunWithDialogue(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	dialogue := &mockDialogue{opening: "你好，请问有什么可以帮您？"}

	s := New(Config{
		SessionID:   "dialogue-test",
		Transport:   trans,
		Transitions: appTransitions(),
		Dialogue:    dialogue,
		Protection: ProtectionConfig{
			MaxDurationSec:         2,
			MaxSilenceSec:          2,
			FirstSilenceTimeoutSec: 1,
		},
	})

	// 关闭音频通道使 Run 退出。
	close(trans.audioCh)

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 检查事件中包含开场白。
	found := false
	for _, evt := range result.Events {
		if evt.EventType == EventBotSpeakStart && evt.Metadata["text"] == "你好，请问有什么可以帮您？" {
			found = true
			break
		}
	}
	assert.True(t, found, "应记录开场白事件")
}

// TestRunWithDialogueAndTTS 验证有 TTS 时开场白异步合成播放。
func TestRunWithDialogueAndTTS(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	tts := &mockTTSProvider{synthData: []byte{1, 2, 3, 4}}
	dialogue := &mockDialogue{opening: "hello"}

	s := New(Config{
		SessionID:   "tts-test",
		Transport:   trans,
		Transitions: appTransitions(),
		Dialogue:    dialogue,
		TTS:         tts,
		Protection: ProtectionConfig{
			MaxDurationSec:         2,
			MaxSilenceSec:          2,
			FirstSilenceTimeoutSec: 1,
		},
	})

	// 给 TTS 一些时间完成，然后关闭通道。
	go func() {
		time.Sleep(200 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// TTS 应该播放了音频。
	played := trans.getPlayedAudio()
	assert.NotEmpty(t, played, "TTS 应播放音频")
}

// ---------------------------------------------------------------------------
// 测试：静默超时流程
// ---------------------------------------------------------------------------

// TestRunSilenceTimeout 验证静默超时触发和二次静默挂断。
func TestRunSilenceTimeout(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	// 使用短超时加速测试。
	s := New(Config{
		SessionID:   "silence-test",
		Transport:   trans,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         10,
			MaxSilenceSec:          1,
			FirstSilenceTimeoutSec: 1,
		},
	})

	// 先驱动 FSM 到 WaitingUser 状态。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 验证记录了静默超时事件。
	silenceCount := 0
	for _, evt := range result.Events {
		if evt.EventType == EventSilenceTimeout {
			silenceCount++
		}
	}
	assert.GreaterOrEqual(t, silenceCount, 2, "应至少记录两次静默超时")
	// FSM 在二次静默后直接进入 Hangup 终态，eventLoop 返回 nil。
}

// ---------------------------------------------------------------------------
// 测试：语音检测和 ASR 流程
// ---------------------------------------------------------------------------

// TestRunSpeechDetectionAndASR 验证语音检测→ASR→对话→TTS完整流程。
func TestRunSpeechDetectionAndASR(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	asrStream := newMockASRStream()
	asr := &mockASRProvider{stream: asrStream}
	tts := &mockTTSProvider{synthData: []byte{1, 2}}
	streamCh := make(chan string, 1)
	dialogue := &mockDialogue{
		opening:  "hello",
		streamCh: streamCh,
	}

	s := New(Config{
		SessionID:       "asr-test",
		Transport:       trans,
		Transitions:     appTransitions(),
		ASR:             asr,
		TTS:             tts,
		Dialogue:        dialogue,
		SpeechDetector:  &mockSpeechDetector{isSpeech: true},
		InputSampleRate: 16000,
		ASRSampleRate:   16000,
		Protection: ProtectionConfig{
			MaxDurationSec:         5,
			MaxSilenceSec:          3,
			FirstSilenceTimeoutSec: 3,
		},
	})

	go func() {
		// 等 Run 开始。
		time.Sleep(100 * time.Millisecond)

		// 发送音频帧以触发语音检测。
		frame := loudFrame(160, 10000)
		trans.audioCh <- frame

		// 模拟 ASR 返回最终结果。
		time.Sleep(50 * time.Millisecond)
		asrStream.eventsCh <- aiface.ASREvent{
			Text:       "你好",
			IsFinal:    true,
			Confidence: 0.95,
		}

		// 发送流式回复。
		time.Sleep(50 * time.Millisecond)
		streamCh <- "你好呀"
		close(streamCh)

		// 等 TTS 播放完成，然后关闭。
		time.Sleep(300 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 验证记录了关键事件。
	eventTypes := make(map[EventType]int)
	for _, evt := range result.Events {
		eventTypes[evt.EventType]++
	}
	assert.Greater(t, eventTypes[EventBotSpeakStart], 0, "应记录 bot_speak_start")
}

// ---------------------------------------------------------------------------
// 测试：Barge-in 流程
// ---------------------------------------------------------------------------

// TestHandleBargeInFrame 验证 barge-in 检测逻辑。
func TestHandleBargeInFrame(t *testing.T) {
	t.Parallel()

	t.Run("TTS未播放时不处理", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "bargein-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: true},
		})
		// ttsPlaying 默认为 false。
		s.ctx = context.Background()

		// 驱动到 BotSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		initialState := s.mfsm.State()
		s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))

		// 状态不应改变。
		assert.Equal(t, initialState, s.mfsm.State())
	})

	t.Run("连续帧不足阈值不触发", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "bargein-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: true},
		})
		s.ctx = context.Background()
		s.ttsPlaying.Store(true)

		// 驱动到 BotSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		// 发送不够阈值的帧。
		for range bargeInThreshold - 1 {
			s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))
		}

		assert.Equal(t, mediafsm.BotSpeaking, s.mfsm.State(), "帧数不足不应触发 barge-in")
	})

	t.Run("达到阈值触发barge-in", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "bargein-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: true},
		})
		s.ctx = context.Background()
		s.ttsPlaying.Store(true)

		// 驱动到 BotSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		for range bargeInThreshold {
			s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))
		}

		// barge-in 后应进入 UserSpeaking（EvBargeIn → BargeIn → EvBargeInDone → UserSpeaking）。
		assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State(), "应触发 barge-in 并进入 UserSpeaking")
		assert.False(t, s.ttsPlaying.Load(), "ttsPlaying 应被清除")
	})

	t.Run("非语音帧重置计数", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "bargein-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: false},
		})
		s.ctx = context.Background()
		s.ttsPlaying.Store(true)
		s.bargeInFrames = 5

		// 驱动到 BotSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		s.handleBargeInFrame(context.Background(), silenceFrame160())
		assert.Equal(t, 0, s.bargeInFrames, "非语音帧应重置计数")
	})
}

// TestHandleBargeInWithTTSCancel 验证 barge-in 取消 TTS。
func TestHandleBargeInWithTTSCancel(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:      "bargein-cancel",
		Transport:      trans,
		Transitions:    appTransitions(),
		SpeechDetector: &mockSpeechDetector{isSpeech: true},
	})
	s.ctx = context.Background()
	s.ttsPlaying.Store(true)

	// 设置 ttsCancel。
	ctx, cancel := context.WithCancel(context.Background())
	s.ttsCancel = cancel

	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	for range bargeInThreshold {
		s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))
	}

	// 验证 ttsCancel 被调用。
	assert.Error(t, ctx.Err(), "TTS context 应被取消")
	assert.Nil(t, s.ttsCancel, "ttsCancel 应被清空")
}

// ---------------------------------------------------------------------------
// 测试：handleSilenceTimeout
// ---------------------------------------------------------------------------

// TestHandleSilenceTimeout 验证静默超时处理。
func TestHandleSilenceTimeout(t *testing.T) {
	t.Parallel()

	t.Run("首次静默超时", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		// 驱动到 WaitingUser → SilenceTimeout。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSilenceTimeout))

		timer := time.NewTimer(time.Hour) // 不会触发。
		defer timer.Stop()

		s.handleSilenceTimeout(timer)

		assert.Equal(t, 1, s.silenceCount)
		// 无 TTS 时，SilenceTimeout → (EvSilencePromptDone) → BotSpeaking → (EvBotDone) → WaitingUser。
		assert.Equal(t, mediafsm.WaitingUser, s.mfsm.State())
	})

	t.Run("二次静默超时触发挂断", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		// 驱动到 SilenceTimeout 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSilenceTimeout))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		// 第一次静默。
		s.handleSilenceTimeout(timer)
		assert.Equal(t, 1, s.silenceCount)

		// 再次驱动到 SilenceTimeout。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSilenceTimeout))

		// 第二次静默。
		s.handleSilenceTimeout(timer)
		assert.Equal(t, 2, s.silenceCount)
		assert.True(t, s.mfsm.IsTerminal(), "二次静默后应进入终态")
	})
}

// ---------------------------------------------------------------------------
// 测试：handleStreamingASR
// ---------------------------------------------------------------------------

// TestHandleStreamingASR 验证流式 ASR 事件处理。
func TestHandleStreamingASR(t *testing.T) {
	t.Parallel()

	t.Run("非最终结果被忽略", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleStreamingASR(context.Background(), aiface.ASREvent{
			Text:    "partial",
			IsFinal: false,
		}, timer)

		s.mu.Lock()
		assert.Empty(t, s.events, "非最终结果不应记录事件")
		s.mu.Unlock()
	})

	t.Run("空文本被忽略", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleStreamingASR(context.Background(), aiface.ASREvent{
			Text:    "",
			IsFinal: true,
		}, timer)

		s.mu.Lock()
		assert.Empty(t, s.events)
		s.mu.Unlock()
	})

	t.Run("非用户状态下丢弃", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()
		// FSM 处于 Idle 状态（非 WaitingUser/UserSpeaking）。

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleStreamingASR(context.Background(), aiface.ASREvent{
			Text:    "hello",
			IsFinal: true,
		}, timer)

		s.mu.Lock()
		assert.Empty(t, s.events, "非用户状态下应丢弃 ASR 结果")
		s.mu.Unlock()
	})

	t.Run("无对话引擎直接完成", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		// 驱动到 UserSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleStreamingASR(context.Background(), aiface.ASREvent{
			Text:       "你好",
			IsFinal:    true,
			Confidence: 0.9,
		}, timer)

		// 应记录 user_speech_end 事件。
		s.mu.Lock()
		found := false
		for _, evt := range s.events {
			if evt.EventType == EventUserSpeechEnd {
				found = true
				assert.Equal(t, "你好", evt.Metadata["text"])
			}
		}
		s.mu.Unlock()
		assert.True(t, found, "应记录 user_speech_end 事件")
	})

	t.Run("对话引擎流式处理失败", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		dialogue := &mockDialogue{
			opening:   "hi",
			streamErr: errors.New("llm failed"),
		}
		s := New(Config{
			SessionID:   "asr-err",
			Transport:   trans,
			Transitions: appTransitions(),
			Dialogue:    dialogue,
			Protection: ProtectionConfig{
				MaxDurationSec:         5,
				MaxSilenceSec:          2,
				FirstSilenceTimeoutSec: 1,
			},
		})
		s.startTime = time.Now()

		// 驱动到 UserSpeaking 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleStreamingASR(context.Background(), aiface.ASREvent{
			Text:    "test",
			IsFinal: true,
		}, timer)

		// 不应 panic，FSM 应处理完成。
	})
}

// ---------------------------------------------------------------------------
// 测试：Filler 音频播放
// ---------------------------------------------------------------------------

// TestPlayFiller 验证填充词播放逻辑。
func TestPlayFiller(t *testing.T) {
	t.Parallel()

	t.Run("无填充词不播放", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))

		s.playFiller(context.Background())
		assert.Empty(t, trans.getPlayedAudio())
	})

	t.Run("播放填充词", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		metrics := &mockMetrics{}
		filler := []byte{1, 2, 3, 4}
		s := New(Config{
			SessionID:    "filler-test",
			Transport:    trans,
			Transitions:  appTransitions(),
			FillerAudios: [][]byte{filler},
			Metrics:      metrics,
		})

		s.playFiller(context.Background())

		played := trans.getPlayedAudio()
		require.Len(t, played, 1)
		assert.Equal(t, filler, played[0])

		metrics.mu.Lock()
		assert.Equal(t, 1, metrics.fillerPlayed, "应记录填充词播放")
		metrics.mu.Unlock()
	})

	t.Run("空填充词数据被跳过", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:    "filler-empty",
			Transport:    trans,
			Transitions:  appTransitions(),
			FillerAudios: [][]byte{{}},
		})

		s.playFiller(context.Background())
		assert.Empty(t, trans.getPlayedAudio())
	})

	t.Run("多个填充词轮询选择", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		fillers := [][]byte{
			{1, 1},
			{2, 2},
			{3, 3},
		}
		s := New(Config{
			SessionID:    "filler-rotate",
			Transport:    trans,
			Transitions:  appTransitions(),
			FillerAudios: fillers,
		})

		// 多次播放验证轮询。
		for range 6 {
			s.playFiller(context.Background())
		}

		played := trans.getPlayedAudio()
		assert.Len(t, played, 6, "应播放6次")
	})

	t.Run("nil Metrics 不 panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:    "filler-nometrics",
			Transport:    trans,
			Transitions:  appTransitions(),
			FillerAudios: [][]byte{{1, 2}},
			// Metrics 默认为 nil。
		})

		assert.NotPanics(t, func() {
			s.playFiller(context.Background())
		})
	})

	t.Run("PlayAudio错误不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		trans.playErr = errors.New("play failed")
		s := New(Config{
			SessionID:    "filler-err",
			Transport:    trans,
			Transitions:  appTransitions(),
			FillerAudios: [][]byte{{1, 2}},
		})

		assert.NotPanics(t, func() {
			s.playFiller(context.Background())
		})
	})
}

// ---------------------------------------------------------------------------
// 测试：startDialogue
// ---------------------------------------------------------------------------

// TestStartDialogue 验证对话启动逻辑。
func TestStartDialogue(t *testing.T) {
	t.Parallel()

	t.Run("nil对话引擎不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
		})
		s.startTime = time.Now()

		assert.NotPanics(t, func() { s.startDialogue() })
	})

	t.Run("无TTS无ASR时直接完成", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		dialogue := &mockDialogue{opening: "welcome"}
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			Dialogue:    dialogue,
		})
		s.startTime = time.Now()

		// 驱动到 Idle → (EvAnswer) → BotSpeaking。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		s.startDialogue()

		// 无 TTS 时应记录 BotSpeakEnd 并尝试 EvBotDone。
		s.mu.Lock()
		hasStart := false
		hasEnd := false
		for _, evt := range s.events {
			if evt.EventType == EventBotSpeakStart {
				hasStart = true
			}
			if evt.EventType == EventBotSpeakEnd {
				hasEnd = true
			}
		}
		s.mu.Unlock()
		assert.True(t, hasStart, "应记录 bot_speak_start")
		assert.True(t, hasEnd, "无 TTS 时应记录 bot_speak_end")
	})
}

// ---------------------------------------------------------------------------
// 测试：startASRStream
// ---------------------------------------------------------------------------

// TestStartASRStream 验证 ASR 流初始化。
func TestStartASRStream(t *testing.T) {
	t.Parallel()

	t.Run("nil ASR不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.ctx = context.Background()

		assert.NotPanics(t, func() { s.startASRStream() })
		assert.Nil(t, s.asrStream)
	})

	t.Run("nil ctx不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		stream := newMockASRStream()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         &mockASRProvider{stream: stream},
		})
		// ctx 默认为 nil。

		assert.NotPanics(t, func() { s.startASRStream() })
		assert.Nil(t, s.asrStream)
	})

	t.Run("ASR流启动失败", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         &mockASRProvider{streamErr: errors.New("asr startup failed")},
		})
		s.ctx = context.Background()

		assert.NotPanics(t, func() { s.startASRStream() })
		assert.Nil(t, s.asrStream)
	})

	t.Run("ASR流启动成功", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		stream := newMockASRStream()
		s := New(Config{
			SessionID:   "test",
			Transport:   trans,
			Transitions: appTransitions(),
			ASR:         &mockASRProvider{stream: stream},
		})
		s.ctx = context.Background()

		s.startASRStream()
		assert.NotNil(t, s.asrStream, "应设置 asrStream")

		// 验证事件转发：向 stream 发送事件，应能从 asrResults 读到。
		stream.eventsCh <- aiface.ASREvent{Text: "test", IsFinal: true}
		select {
		case evt := <-s.asrResults:
			assert.Equal(t, "test", evt.Text)
		case <-time.After(time.Second):
			t.Fatal("ASR 事件转发超时")
		}
	})
}

// ---------------------------------------------------------------------------
// 测试：synthesizeAndPlayAsync
// ---------------------------------------------------------------------------

// TestSynthesizeAndPlayAsync 验证异步 TTS 合成播放。
func TestSynthesizeAndPlayAsync(t *testing.T) {
	t.Parallel()

	t.Run("合成并播放", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: []byte{10, 20, 30}}
		s := New(Config{
			SessionID:   "synth-test",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		s.synthesizeAndPlayAsync("hello")

		// 等待 botDoneCh 信号。
		select {
		case <-s.botDoneCh:
			// 成功。
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		played := trans.getPlayedAudio()
		require.Len(t, played, 1)
		assert.Equal(t, []byte{10, 20, 30}, played[0])
	})

	t.Run("合成失败", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthErr: errors.New("tts failed")}
		s := New(Config{
			SessionID:   "synth-err",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		s.synthesizeAndPlayAsync("hello")

		select {
		case <-s.botDoneCh:
			// 即使失败也应发送完成信号。
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		assert.Empty(t, trans.getPlayedAudio(), "合成失败不应播放")
	})

	t.Run("空音频数据不播放", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: nil}
		s := New(Config{
			SessionID:   "synth-empty",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		s.synthesizeAndPlayAsync("hello")

		select {
		case <-s.botDoneCh:
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		assert.Empty(t, trans.getPlayedAudio())
	})

	t.Run("取消前一个TTS", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: []byte{1}}
		s := New(Config{
			SessionID:   "synth-cancel",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		// 设置旧的 ttsCancel。
		oldCtx, oldCancel := context.WithCancel(context.Background())
		s.ttsCancel = oldCancel

		s.synthesizeAndPlayAsync("new text")

		// 旧的 context 应被取消。
		assert.Error(t, oldCtx.Err(), "旧 TTS context 应被取消")

		select {
		case <-s.botDoneCh:
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}
	})
}

// ---------------------------------------------------------------------------
// 测试：synthesizeAndPlayStreamAsync
// ---------------------------------------------------------------------------

// TestSynthesizeAndPlayStreamAsync 验证流式 TTS 合成播放。
func TestSynthesizeAndPlayStreamAsync(t *testing.T) {
	t.Parallel()

	t.Run("流式合成多个句段", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: []byte{1, 2}}
		s := New(Config{
			SessionID:   "stream-synth",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		sentenceCh := make(chan string, 3)
		sentenceCh <- "第一句"
		sentenceCh <- "第二句"
		close(sentenceCh)

		s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

		select {
		case <-s.botDoneCh:
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		played := trans.getPlayedAudio()
		assert.GreaterOrEqual(t, len(played), 2, "应播放至少2个句段")
	})

	t.Run("带onComplete回调", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: []byte{1}}
		s := New(Config{
			SessionID:   "stream-complete",
			Transport:   trans,
			Transitions: appTransitions(),
			TTS:         tts,
		})
		s.ctx = context.Background()

		sentenceCh := make(chan string, 1)
		sentenceCh <- "text"
		close(sentenceCh)

		var completed atomic.Bool
		s.synthesizeAndPlayStreamAsync(sentenceCh, func() {
			completed.Store(true)
		})

		select {
		case <-s.botDoneCh:
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		assert.True(t, completed.Load(), "onComplete 应被调用")
	})

	t.Run("带填充词播放", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		tts := &mockTTSProvider{synthData: []byte{1}}
		metrics := &mockMetrics{}
		s := New(Config{
			SessionID:    "stream-filler",
			Transport:    trans,
			Transitions:  appTransitions(),
			TTS:          tts,
			Metrics:      metrics,
			FillerAudios: [][]byte{{99, 99}},
		})
		s.ctx = context.Background()

		sentenceCh := make(chan string, 1)
		sentenceCh <- "text"
		close(sentenceCh)

		s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

		select {
		case <-s.botDoneCh:
		case <-time.After(2 * time.Second):
			t.Fatal("botDoneCh 超时")
		}

		played := trans.getPlayedAudio()
		// 第一个应该是填充词，后面是合成音频。
		assert.GreaterOrEqual(t, len(played), 2, "应播放填充词和合成音频")

		metrics.mu.Lock()
		assert.Equal(t, 1, metrics.fillerPlayed, "应记录填充词播放")
		metrics.mu.Unlock()
	})
}

// ---------------------------------------------------------------------------
// 测试：handleHangup
// ---------------------------------------------------------------------------

// TestHandleHangup 验证挂断处理。
func TestHandleHangup(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()

	// 驱动到 WaitingUser 状态。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

	s.handleHangup("user_disconnect")

	assert.Equal(t, mediafsm.Hangup, s.mfsm.State())

	s.mu.Lock()
	found := false
	for _, evt := range s.events {
		if evt.EventType == EventHangup && evt.Metadata["cause"] == "user_disconnect" {
			found = true
		}
	}
	s.mu.Unlock()
	assert.True(t, found, "应记录挂断事件")
}

// ---------------------------------------------------------------------------
// 测试：handleWaitingUserFrame
// ---------------------------------------------------------------------------

// TestHandleWaitingUserFrame 验证等待用户说话时的帧处理。
func TestHandleWaitingUserFrame(t *testing.T) {
	t.Parallel()

	t.Run("非语音帧不触发", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "wait-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: false},
		})
		s.startTime = time.Now()

		// 驱动到 WaitingUser 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleWaitingUserFrame(silenceFrame160(), timer)
		assert.Equal(t, mediafsm.WaitingUser, s.mfsm.State(), "静音帧不应改变状态")
	})

	t.Run("语音帧触发SpeechStart", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "wait-test",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: true},
		})
		s.startTime = time.Now()

		// 驱动到 WaitingUser 状态。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		s.handleWaitingUserFrame(loudFrame(160, 10000), timer)
		assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State(), "语音帧应触发 SpeechStart")
	})
}

// ---------------------------------------------------------------------------
// 测试：handleAudioFrame
// ---------------------------------------------------------------------------

// TestHandleAudioFrame 验证音频帧分发逻辑。
func TestHandleAudioFrame(t *testing.T) {
	t.Parallel()

	t.Run("Idle状态不处理", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(minimalConfig(trans))
		s.startTime = time.Now()

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		// Idle 状态，不应 panic。
		assert.NotPanics(t, func() {
			s.handleAudioFrame(context.Background(), loudFrame(160, 10000), timer)
		})
	})

	t.Run("WaitingUser状态送入ASR并检测语音", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		trans := newMockTransport()
		s := New(Config{
			SessionID:       "audio-test",
			Transport:       trans,
			Transitions:     appTransitions(),
			SpeechDetector:  &mockSpeechDetector{isSpeech: true},
			InputSampleRate: 16000,
			ASRSampleRate:   16000,
		})
		s.startTime = time.Now()
		s.asrStream = stream

		// 驱动到 WaitingUser。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		frame := loudFrame(160, 10000)
		s.handleAudioFrame(context.Background(), frame, timer)

		fed := stream.getFedData()
		assert.Len(t, fed, 1, "应送入 ASR")
		assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State())
	})

	t.Run("UserSpeaking状态送入ASR", func(t *testing.T) {
		t.Parallel()
		stream := newMockASRStream()
		trans := newMockTransport()
		s := New(Config{
			SessionID:       "audio-test",
			Transport:       trans,
			Transitions:     appTransitions(),
			InputSampleRate: 16000,
			ASRSampleRate:   16000,
		})
		s.startTime = time.Now()
		s.asrStream = stream

		// 驱动到 UserSpeaking。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
		require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		frame := loudFrame(160, 10000)
		s.handleAudioFrame(context.Background(), frame, timer)

		fed := stream.getFedData()
		assert.Len(t, fed, 1, "UserSpeaking 应继续送入 ASR")
	})

	t.Run("BotSpeaking状态检测打断", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:      "audio-barge",
			Transport:      trans,
			Transitions:    appTransitions(),
			SpeechDetector: &mockSpeechDetector{isSpeech: true},
		})
		s.startTime = time.Now()
		s.ctx = context.Background()
		s.ttsPlaying.Store(true)

		// 驱动到 BotSpeaking。
		require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		// 发送足够帧触发 barge-in。
		for range bargeInThreshold {
			s.handleAudioFrame(context.Background(), loudFrame(160, 10000), timer)
		}

		assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State(), "应触发 barge-in")
	})
}

// ---------------------------------------------------------------------------
// 测试：handleFSMEvent 和 tryHandleFSMEvent
// ---------------------------------------------------------------------------

// TestHandleFSMEvent 验证 FSM 事件处理。
func TestHandleFSMEvent(t *testing.T) {
	t.Parallel()

	t.Run("有效事件转换", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "fsm-test",
			Transport:   trans,
			Transitions: appTransitions(),
		})

		s.handleFSMEvent(mediafsm.EvAnswer, "test answer")
		assert.Equal(t, mediafsm.BotSpeaking, s.mfsm.State())
	})

	t.Run("无效事件不panic", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "fsm-test",
			Transport:   trans,
			Transitions: appTransitions(),
		})

		// Idle 状态下发送 EvBotDone 是无效的。
		assert.NotPanics(t, func() {
			s.handleFSMEvent(mediafsm.EvBotDone, "invalid event")
		})
	})
}

// TestTryHandleFSMEvent 验证 tryHandleFSMEvent 仅在可处理时发送。
func TestTryHandleFSMEvent(t *testing.T) {
	t.Parallel()

	t.Run("可处理事件发送", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "try-fsm",
			Transport:   trans,
			Transitions: appTransitions(),
		})

		s.tryHandleFSMEvent(mediafsm.EvAnswer, "try answer")
		assert.Equal(t, mediafsm.BotSpeaking, s.mfsm.State())
	})

	t.Run("不可处理事件被跳过", func(t *testing.T) {
		t.Parallel()
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "try-fsm",
			Transport:   trans,
			Transitions: appTransitions(),
		})

		// Idle 状态下 EvBotDone 不可处理。
		s.tryHandleFSMEvent(mediafsm.EvBotDone, "skip")
		assert.Equal(t, mediafsm.Idle, s.mfsm.State(), "不可处理事件不应改变状态")
	})
}

// ---------------------------------------------------------------------------
// 测试：完整 Run() 集成流程
// ---------------------------------------------------------------------------

// TestRunFullFlow 验证 Run() 的完整对话流程（无 TTS）。
func TestRunFullFlow(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	asrStream := newMockASRStream()
	dialogue := &mockDialogue{
		opening: "你好",
	}

	s := New(Config{
		SessionID:   "full-flow",
		Transport:   trans,
		Transitions: appTransitions(),
		ASR:         &mockASRProvider{stream: asrStream},
		Dialogue:    dialogue,
		Protection: ProtectionConfig{
			MaxDurationSec:         5,
			MaxSilenceSec:          2,
			FirstSilenceTimeoutSec: 1,
		},
	})

	// 先驱动到 BotSpeaking（模拟 APP 场景会话建立）。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	go func() {
		// 等 Run 开始，然后关闭音频通道。
		time.Sleep(200 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "full-flow", result.SessionID)
}

// TestRunStreamingASRWithDialogueAndTTS 验证完整的 ASR→Dialogue→TTS 流式管道。
func TestRunStreamingASRWithDialogueAndTTS(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	asrStream := newMockASRStream()
	asr := &mockASRProvider{stream: asrStream}
	tts := &mockTTSProvider{synthData: []byte{1, 2, 3}}

	streamCh := make(chan string, 2)
	dialogue := &mockDialogue{
		opening:  "你好",
		streamCh: streamCh,
	}

	s := New(Config{
		SessionID:       "stream-full",
		Transport:       trans,
		Transitions:     appTransitions(),
		ASR:             asr,
		TTS:             tts,
		Dialogue:        dialogue,
		InputSampleRate: 16000,
		ASRSampleRate:   16000,
		SpeechDetector:  &mockSpeechDetector{isSpeech: true},
		Protection: ProtectionConfig{
			MaxDurationSec:         10,
			MaxSilenceSec:          5,
			FirstSilenceTimeoutSec: 5,
		},
	})

	// APP 场景：先驱动 FSM 到 BotSpeaking（模拟会话建立）。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	go func() {
		// 等 Run 启动、TTS 开场白合成播放完成，以及 botDoneCh 被处理后 FSM 进入 WaitingUser。
		time.Sleep(500 * time.Millisecond)

		// 发送音频帧触发语音开始。
		trans.audioCh <- loudFrame(160, 10000)

		// 给一点时间让 FSM 转换到 UserSpeaking。
		time.Sleep(100 * time.Millisecond)

		// 模拟 ASR 最终结果。
		asrStream.eventsCh <- aiface.ASREvent{
			Text:       "帮我查一下天气",
			IsFinal:    true,
			Confidence: 0.92,
		}

		// 等对话引擎处理，然后发送流式回复。
		time.Sleep(200 * time.Millisecond)
		streamCh <- "今天天气晴朗"
		streamCh <- "最高温度25度"
		close(streamCh)

		// 等 TTS 播放完成后关闭。
		time.Sleep(500 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 验证事件记录。
	eventTypes := make(map[EventType]int)
	for _, evt := range result.Events {
		eventTypes[evt.EventType]++
	}
	assert.Greater(t, eventTypes[EventBotSpeakStart], 0, "应记录 bot_speak_start")
	assert.Greater(t, eventTypes[EventUserSpeechEnd], 0, "应记录 user_speech_end")
}

// TestSynthesizeAndPlayStreamAsyncContextCancelled 验证流式合成中 context 取消。
func TestSynthesizeAndPlayStreamAsyncContextCancelled(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	tts := &mockTTSProvider{synthData: []byte{1, 2}}

	parentCtx, parentCancel := context.WithCancel(context.Background())
	s := New(Config{
		SessionID:   "stream-cancel",
		Transport:   trans,
		Transitions: appTransitions(),
		TTS:         tts,
	})
	s.ctx = parentCtx

	sentenceCh := make(chan string, 10)
	sentenceCh <- "第一句"

	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

	// 等第一句处理完。
	time.Sleep(100 * time.Millisecond)

	// 取消 parent context，模拟 barge-in 或超时。
	parentCancel()

	// 再发一句（应被跳过）。
	sentenceCh <- "第二句"
	close(sentenceCh)

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("botDoneCh 超时")
	}
}

// TestSynthesizeAndPlayStreamAsyncTTSSynthError 验证流式合成中 TTS 合成失败继续处理。
func TestSynthesizeAndPlayStreamAsyncTTSSynthError(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	tts := &mockTTSProvider{synthErr: errors.New("synth error")}
	s := New(Config{
		SessionID:   "stream-err",
		Transport:   trans,
		Transitions: appTransitions(),
		TTS:         tts,
	})
	s.ctx = context.Background()

	sentenceCh := make(chan string, 2)
	sentenceCh <- "句子一"
	sentenceCh <- "句子二"
	close(sentenceCh)

	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("botDoneCh 超时")
	}

	// 合成失败不应有播放。
	assert.Empty(t, trans.getPlayedAudio())
}

// TestSynthesizeAndPlayStreamAsyncEmptyAudio 验证流式合成返回空音频跳过。
func TestSynthesizeAndPlayStreamAsyncEmptyAudio(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	tts := &mockTTSProvider{synthData: nil} // 返回空音频。
	s := New(Config{
		SessionID:   "stream-empty",
		Transport:   trans,
		Transitions: appTransitions(),
		TTS:         tts,
	})
	s.ctx = context.Background()

	sentenceCh := make(chan string, 1)
	sentenceCh <- "text"
	close(sentenceCh)

	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("botDoneCh 超时")
	}

	assert.Empty(t, trans.getPlayedAudio())
}

// TestSynthesizeAndPlayAsyncContextCancelled 验证异步合成中 context 被取消。
func TestSynthesizeAndPlayAsyncContextCancelled(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	trans.playBlocking = true // 模拟播放阻塞直到 ctx 取消。

	tts := &mockTTSProvider{synthData: []byte{1, 2, 3}}
	parentCtx, parentCancel := context.WithCancel(context.Background())
	s := New(Config{
		SessionID:   "async-cancel",
		Transport:   trans,
		Transitions: appTransitions(),
		TTS:         tts,
	})
	s.ctx = parentCtx

	s.synthesizeAndPlayAsync("text")

	// 等 TTS 合成完成，开始播放。
	time.Sleep(100 * time.Millisecond)
	parentCancel()

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("botDoneCh 超时")
	}
}

// TestSynthesizeAndPlayStreamAsyncPlayError 验证流式播放失败时继续。
func TestSynthesizeAndPlayStreamAsyncPlayError(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	trans.playErr = errors.New("play error")
	tts := &mockTTSProvider{synthData: []byte{1, 2}}
	s := New(Config{
		SessionID:   "stream-play-err",
		Transport:   trans,
		Transitions: appTransitions(),
		TTS:         tts,
	})
	s.ctx = context.Background()

	sentenceCh := make(chan string, 2)
	sentenceCh <- "a"
	sentenceCh <- "b"
	close(sentenceCh)

	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("botDoneCh 超时")
	}
}

// TestHandleBargeInStopPlaybackError 验证 barge-in 时 StopPlayback 错误不中断。
func TestHandleBargeInStopPlaybackError(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	trans.stopErr = errors.New("stop error")
	s := New(Config{
		SessionID:      "bargein-stop-err",
		Transport:      trans,
		Transitions:    appTransitions(),
		SpeechDetector: &mockSpeechDetector{isSpeech: true},
	})
	s.ctx = context.Background()
	s.ttsPlaying.Store(true)

	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	for range bargeInThreshold {
		s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))
	}

	// 即使 StopPlayback 失败也应完成 barge-in。
	assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State())
}

// TestHandleStreamingASRWithNoTTS 验证有对话引擎但无 TTS 时的流式处理。
func TestHandleStreamingASRWithNoTTS(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	streamCh := make(chan string, 1)
	streamCh <- "reply"
	close(streamCh)

	dialogue := &mockDialogue{
		opening:  "hi",
		streamCh: streamCh,
	}
	s := New(Config{
		SessionID:   "no-tts-stream",
		Transport:   trans,
		Transitions: appTransitions(),
		Dialogue:    dialogue,
		Protection: ProtectionConfig{
			MaxDurationSec:         5,
			MaxSilenceSec:          2,
			FirstSilenceTimeoutSec: 1,
		},
	})
	s.startTime = time.Now()

	// 驱动到 UserSpeaking。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()

	s.handleStreamingASR(context.Background(), aiface.ASREvent{
		Text:    "test",
		IsFinal: true,
	}, timer)

	// 无 TTS 时应直接处理完成。
	// 验证不 panic 且事件被记录。
	s.mu.Lock()
	hasEnd := false
	for _, evt := range s.events {
		if evt.EventType == EventUserSpeechEnd {
			hasEnd = true
		}
	}
	s.mu.Unlock()
	assert.True(t, hasEnd)
}

// TestStartASRStreamEventForwarding 验证 ASR 事件通道满时丢弃。
func TestStartASRStreamEventForwarding(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	stream := newMockASRStream()
	s := New(Config{
		SessionID:   "asr-overflow",
		Transport:   trans,
		Transitions: appTransitions(),
		ASR:         &mockASRProvider{stream: stream},
	})
	s.ctx = context.Background()

	s.startASRStream()

	// 填满 asrResults 通道（容量 16）。
	for range 16 {
		stream.eventsCh <- aiface.ASREvent{Text: "full", IsFinal: false}
	}

	// 等事件被转发。
	time.Sleep(100 * time.Millisecond)

	// 再发一个，应被丢弃不阻塞。
	stream.eventsCh <- aiface.ASREvent{Text: "overflow", IsFinal: false}
	time.Sleep(50 * time.Millisecond)

	// 关闭事件通道结束转发 goroutine。
	close(stream.eventsCh)
}

// TestRunMaxDuration 验证最大时长保护。
func TestRunMaxDuration(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:   "max-dur",
		Transport:   trans,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         1,
			MaxSilenceSec:          30,
			FirstSilenceTimeoutSec: 30,
		},
	})

	start := time.Now()
	result, err := s.Run(context.Background())
	elapsed := time.Since(start)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "context")
	require.NotNil(t, result)
	// 应在大约 1 秒内超时。
	assert.Less(t, elapsed, 3*time.Second, "应在超时后退出")
}

// ---------------------------------------------------------------------------
// 补充覆盖率测试
// ---------------------------------------------------------------------------

// ctxAwareTTS 感知 context 取消的 TTS mock。
type ctxAwareTTS struct {
	synthData []byte
	synthErr  error
}

func (m *ctxAwareTTS) Synthesize(ctx context.Context, _ string, _ aiface.TTSConfig) ([]byte, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if m.synthErr != nil {
		return nil, m.synthErr
	}
	return m.synthData, nil
}

func (m *ctxAwareTTS) SynthesizeStream(_ context.Context, textCh <-chan string, _ aiface.TTSConfig) (<-chan []byte, error) {
	ch := make(chan []byte, 8)
	go func() {
		defer close(ch)
		for range textCh {
			if m.synthData != nil {
				ch <- m.synthData
			}
		}
	}()
	return ch, nil
}

func (m *ctxAwareTTS) Cancel() error { return nil }

func TestSynthesizeAndPlayAsync_Cancelled(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	tts := &ctxAwareTTS{synthData: []byte{1, 2}}
	s := New(Config{
		SessionID: "cancel-ctx", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	ctx, cancel := context.WithCancel(context.Background())
	s.ctx = ctx
	cancel()
	s.synthesizeAndPlayAsync("text")
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
	assert.Empty(t, trans.getPlayedAudio())
}

func TestSynthesizeAndPlayAsync_PlayError(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	trans.playErr = errors.New("play failed")
	tts := &ctxAwareTTS{synthData: []byte{1, 2}}
	s := New(Config{
		SessionID: "play-err", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	s.ctx = context.Background()
	s.synthesizeAndPlayAsync("text")
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
}

func TestSynthesizeAndPlayStreamAsync_SynthError2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	tts := &mockTTSProvider{synthErr: errors.New("synth error")}
	s := New(Config{
		SessionID: "synth-err2", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	s.ctx = context.Background()
	sentenceCh := make(chan string, 2)
	sentenceCh <- "第一句"
	sentenceCh <- "第二句"
	close(sentenceCh)
	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
}

func TestSynthesizeAndPlayStreamAsync_EmptyAudio2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	tts := &mockTTSProvider{synthData: nil}
	s := New(Config{
		SessionID: "empty-audio2", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	s.ctx = context.Background()
	sentenceCh := make(chan string, 1)
	sentenceCh <- "text"
	close(sentenceCh)
	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
	assert.Empty(t, trans.getPlayedAudio())
}

func TestSynthesizeAndPlayStreamAsync_PlayError2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	trans.playErr = errors.New("play error")
	tts := &mockTTSProvider{synthData: []byte{1, 2}}
	s := New(Config{
		SessionID: "play-err2", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	s.ctx = context.Background()
	sentenceCh := make(chan string, 2)
	sentenceCh <- "t1"
	sentenceCh <- "t2"
	close(sentenceCh)
	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
}

func TestSynthesizeAndPlayStreamAsync_CtxCancelled2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	tts := &ctxAwareTTS{synthData: []byte{1, 2}}
	s := New(Config{
		SessionID: "ctx-cancel2", Transport: trans, Transitions: appTransitions(), TTS: tts,
	})
	ctx, cancel := context.WithCancel(context.Background())
	s.ctx = ctx
	cancel()
	sentenceCh := make(chan string, 3)
	sentenceCh <- "t1"
	sentenceCh <- "t2"
	close(sentenceCh)
	s.synthesizeAndPlayStreamAsync(sentenceCh, nil)
	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
}

func TestHandleStreamingASR_WithTTSStream(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	tts := &mockTTSProvider{synthData: []byte{1, 2}}
	streamCh := make(chan string, 2)
	streamCh <- "回复一"
	streamCh <- "回复二"
	close(streamCh)
	dialogue := &mockDialogue{opening: "你好", streamCh: streamCh}

	s := New(Config{
		SessionID: "asr-tts", Transport: trans, Transitions: appTransitions(),
		TTS: tts, Dialogue: dialogue,
	})
	s.startTime = time.Now()
	s.ctx = context.Background()
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	s.handleStreamingASR(context.Background(), aiface.ASREvent{
		Text: "你好", IsFinal: true,
	}, timer)

	select {
	case <-s.botDoneCh:
	case <-time.After(2 * time.Second):
		t.Fatal("超时")
	}
	assert.NotEmpty(t, trans.getPlayedAudio())
}

func TestHandleStreamingASR_NoTTSWithDialogue2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	streamCh := make(chan string, 1)
	streamCh <- "回复"
	close(streamCh)
	dialogue := &mockDialogue{opening: "你好", streamCh: streamCh}

	s := New(Config{
		SessionID: "no-tts2", Transport: trans, Transitions: appTransitions(),
		Dialogue: dialogue,
	})
	s.startTime = time.Now()
	s.ctx = context.Background()
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvSpeechStart))

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	s.handleStreamingASR(context.Background(), aiface.ASREvent{
		Text: "测试", IsFinal: true,
	}, timer)
	time.Sleep(100 * time.Millisecond)
}

func TestHandleBargeInFrame_StopPlaybackError2(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	trans.stopErr = errors.New("stop error")
	s := New(Config{
		SessionID: "barge-err2", Transport: trans, Transitions: appTransitions(),
		SpeechDetector: &mockSpeechDetector{isSpeech: true},
	})
	s.startTime = time.Now()
	s.ctx = context.Background()
	s.ttsPlaying.Store(true)
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	for range bargeInThreshold {
		s.handleBargeInFrame(context.Background(), loudFrame(160, 10000))
	}
	assert.Equal(t, mediafsm.UserSpeaking, s.mfsm.State())
}

func TestStartASRStream_SuccessForward(t *testing.T) {
	t.Parallel()
	trans := newMockTransport()
	stream := newMockASRStream()
	s := New(Config{
		SessionID: "asr-fwd", Transport: trans, Transitions: appTransitions(),
		ASR: &mockASRProvider{stream: stream},
	})
	s.ctx = context.Background()
	s.startASRStream()
	assert.NotNil(t, s.asrStream)
	stream.eventsCh <- aiface.ASREvent{Text: "test", IsFinal: true}
	select {
	case evt := <-s.asrResults:
		assert.Equal(t, "test", evt.Text)
	case <-time.After(time.Second):
		t.Fatal("未转发")
	}
}

// ---------------------------------------------------------------------------
// 基准测试
// ---------------------------------------------------------------------------

// BenchmarkNew 基准测试 Session 创建。
func BenchmarkNew(b *testing.B) {
	trans := newMockTransport()
	cfg := Config{
		SessionID:   "bench",
		Transport:   trans,
		Transitions: appTransitions(),
	}
	b.ResetTimer()
	for range b.N {
		_ = New(cfg)
	}
}

// BenchmarkRecordEvent 基准测试事件记录。
func BenchmarkRecordEvent(b *testing.B) {
	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()
	metadata := map[string]string{"key": "value"}

	b.ResetTimer()
	for range b.N {
		s.RecordEvent(EventHangup, metadata)
	}
}

// BenchmarkIsSpeechFrame 基准测试语音检测热路径。
func BenchmarkIsSpeechFrame(b *testing.B) {
	trans := newMockTransport()
	detector := &mockSpeechDetector{isSpeech: true}
	s := New(Config{
		SessionID:      "bench",
		Transport:      trans,
		Transitions:    appTransitions(),
		SpeechDetector: detector,
	})
	frame := loudFrame(160, 10000)

	b.ResetTimer()
	for range b.N {
		_ = s.isSpeechFrame(frame)
	}
}

// BenchmarkIsSpeechFrameEnergyFallback 基准测试能量退回检测。
func BenchmarkIsSpeechFrameEnergyFallback(b *testing.B) {
	trans := newMockTransport()
	s := New(Config{
		SessionID:   "bench",
		Transport:   trans,
		Transitions: appTransitions(),
		// 无 SpeechDetector，使用能量检测退回方案。
	})
	frame := loudFrame(160, 10000)

	b.ResetTimer()
	for range b.N {
		_ = s.isSpeechFrame(frame)
	}
}

// BenchmarkBuildResult 基准测试结果构建。
func BenchmarkBuildResult(b *testing.B) {
	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()
	for range 100 {
		s.RecordEvent(EventHangup, map[string]string{"k": "v"})
	}

	b.ResetTimer()
	for range b.N {
		_ = s.buildResult()
	}
}

// BenchmarkHandleAudioFrame 基准测试音频帧处理热路径。
func BenchmarkHandleAudioFrame(b *testing.B) {
	trans := newMockTransport()
	stream := newMockASRStream()
	s := New(Config{
		SessionID:       "bench",
		Transport:       trans,
		Transitions:     appTransitions(),
		SpeechDetector:  &mockSpeechDetector{isSpeech: false},
		InputSampleRate: 16000,
		ASRSampleRate:   16000,
	})
	s.startTime = time.Now()
	s.asrStream = stream
	// 驱动到 WaitingUser 状态。
	_ = s.mfsm.Handle(mediafsm.EvAnswer)
	_ = s.mfsm.Handle(mediafsm.EvBotDone)

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	frame := silenceFrame160()

	b.ResetTimer()
	for range b.N {
		s.handleAudioFrame(context.Background(), frame, timer)
	}
}

// BenchmarkRecordEventConcurrent 基准测试并发事件记录。
func BenchmarkRecordEventConcurrent(b *testing.B) {
	trans := newMockTransport()
	s := New(minimalConfig(trans))
	s.startTime = time.Now()
	metadata := map[string]string{"key": "value"}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			s.RecordEvent(EventHangup, metadata)
		}
	})
}

// ---------------------------------------------------------------------------
// 模糊测试
// ---------------------------------------------------------------------------

// FuzzIsSpeechFrame 模糊测试语音检测不会 panic。
func FuzzIsSpeechFrame(f *testing.F) {
	f.Add(silenceFrame160())
	f.Add(loudFrame(160, 10000))
	f.Add(loudFrame(160, 32767))
	f.Add([]byte{})
	f.Add([]byte{1})
	f.Add(loudFrame(1, 100))

	f.Fuzz(func(t *testing.T, frame []byte) {
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "fuzz",
			Transport:   trans,
			Transitions: appTransitions(),
		})
		// 不应 panic。
		_ = s.isSpeechFrame(frame)
	})
}

// FuzzRecordEvent 模糊测试事件记录不会 panic。
func FuzzRecordEvent(f *testing.F) {
	f.Add("hangup", "cause", "normal")
	f.Add("", "", "")
	f.Add("user_speech_start", "text", "你好")
	f.Add("中文事件", "中文键", "中文值")

	f.Fuzz(func(t *testing.T, eventType, metaKey, metaValue string) {
		trans := newMockTransport()
		s := New(Config{
			SessionID:   "fuzz",
			Transport:   trans,
			Transitions: appTransitions(),
		})
		s.startTime = time.Now()

		metadata := map[string]string{metaKey: metaValue}
		s.RecordEvent(EventType(eventType), metadata)

		s.mu.Lock()
		if len(s.events) != 1 {
			t.Fatal("应记录1个事件")
		}
		evt := s.events[0]
		s.mu.Unlock()

		if string(evt.EventType) != eventType {
			t.Errorf("事件类型不匹配: got %s, want %s", evt.EventType, eventType)
		}
	})
}

// FuzzNewConfig 模糊测试 New() 各种配置不会 panic。
func FuzzNewConfig(f *testing.F) {
	f.Add("session-1", 8000, 16000, -35.0, 300, 15, 6)
	f.Add("", 0, 0, 0.0, 0, 0, 0)
	f.Add("s", -1, -1, -100.0, -1, -1, -1)
	f.Add("中文ID", 44100, 44100, 10.0, 9999, 9999, 9999)

	f.Fuzz(func(t *testing.T, sessionID string, inputRate, asrRate int, threshold float64, maxDur, maxSil, firstSil int) {
		trans := newMockTransport()
		s := New(Config{
			SessionID:           sessionID,
			Transport:           trans,
			Transitions:         appTransitions(),
			InputSampleRate:     inputRate,
			ASRSampleRate:       asrRate,
			EnergyThresholdDBFS: threshold,
			Protection: ProtectionConfig{
				MaxDurationSec:         maxDur,
				MaxSilenceSec:          maxSil,
				FirstSilenceTimeoutSec: firstSil,
			},
		})
		if s == nil {
			t.Fatal("New() 不应返回 nil")
		}
	})
}

// ---------------------------------------------------------------------------
// 集成测试：完整 E2E 流程
// ---------------------------------------------------------------------------

// TestE2EAppFlowWithBargeIn 模拟 APP 场景完整对话流程（含打断）。
func TestE2EAppFlowWithBargeIn(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	asrStream := newMockASRStream()
	asr := &mockASRProvider{stream: asrStream}
	tts := &mockTTSProvider{synthData: []byte{1, 2, 3}}

	streamCh1 := make(chan string, 2)
	dialogue := &mockDialogue{
		opening:  "欢迎使用",
		streamCh: streamCh1,
	}

	metrics := &mockMetrics{}

	s := New(Config{
		SessionID:       "e2e-bargein",
		Transport:       trans,
		Transitions:     appTransitions(),
		ASR:             asr,
		TTS:             tts,
		Dialogue:        dialogue,
		Metrics:         metrics,
		SpeechDetector:  &mockSpeechDetector{isSpeech: true},
		InputSampleRate: 16000,
		ASRSampleRate:   16000,
		Protection: ProtectionConfig{
			MaxDurationSec:         10,
			MaxSilenceSec:          5,
			FirstSilenceTimeoutSec: 5,
		},
	})

	// APP 场景先驱动到 BotSpeaking。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	go func() {
		// 等 Run 启动和开场白 TTS 播放。
		time.Sleep(500 * time.Millisecond)

		// 发送用户语音，触发 SpeechStart。
		trans.audioCh <- loudFrame(160, 10000)
		time.Sleep(50 * time.Millisecond)

		// ASR 最终结果。
		asrStream.eventsCh <- aiface.ASREvent{
			Text: "我想查天气", IsFinal: true, Confidence: 0.9,
		}

		// 对话引擎回复。
		time.Sleep(100 * time.Millisecond)
		streamCh1 <- "今天天气很好"
		close(streamCh1)

		// 等 TTS 播放，然后正常关闭。
		time.Sleep(500 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "e2e-bargein", result.SessionID)
	assert.NotEmpty(t, result.Events)
}

// TestE2ESilenceEscalation 模拟静默升级到挂断的完整流程。
func TestE2ESilenceEscalation(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()

	s := New(Config{
		SessionID:   "e2e-silence",
		Transport:   trans,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         10,
			MaxSilenceSec:          1,
			FirstSilenceTimeoutSec: 1,
		},
	})

	// 驱动到 WaitingUser 状态。
	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))
	require.NoError(t, s.mfsm.Handle(mediafsm.EvBotDone))

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 验证静默升级：二次静默后进入终态。
	silenceCount := 0
	for _, evt := range result.Events {
		if evt.EventType == EventSilenceTimeout {
			silenceCount++
		}
	}
	assert.GreaterOrEqual(t, silenceCount, 2, "应至少两次静默超时")
}

// TestE2EMaxDurationProtection 验证最大时长保护自动挂断。
func TestE2EMaxDurationProtection(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	s := New(Config{
		SessionID:   "e2e-maxdur",
		Transport:   trans,
		Transitions: appTransitions(),
		Protection: ProtectionConfig{
			MaxDurationSec:         1,
			MaxSilenceSec:          30,
			FirstSilenceTimeoutSec: 30,
		},
	})

	start := time.Now()
	result, err := s.Run(context.Background())
	elapsed := time.Since(start)

	require.Error(t, err, "应因超时返回错误")
	require.NotNil(t, result)
	assert.Less(t, elapsed, 3*time.Second, "应在约1秒后退出")

	hasHangup := false
	for _, evt := range result.Events {
		if evt.EventType == EventHangup {
			hasHangup = true
			assert.Equal(t, "max_duration", evt.Metadata["cause"])
		}
	}
	assert.True(t, hasHangup, "应记录 max_duration 挂断事件")
}

// TestE2EMultiTurnConversation 模拟多轮对话。
func TestE2EMultiTurnConversation(t *testing.T) {
	t.Parallel()

	trans := newMockTransport()
	asrStream := newMockASRStream()
	tts := &mockTTSProvider{synthData: []byte{10, 20}}
	dialogue := &mockDialogue{opening: "你好"}

	s := New(Config{
		SessionID:       "e2e-multi",
		Transport:       trans,
		Transitions:     appTransitions(),
		ASR:             &mockASRProvider{stream: asrStream},
		TTS:             tts,
		Dialogue:        dialogue,
		SpeechDetector:  &mockSpeechDetector{isSpeech: true},
		InputSampleRate: 16000,
		ASRSampleRate:   16000,
		Protection: ProtectionConfig{
			MaxDurationSec:         10,
			MaxSilenceSec:          5,
			FirstSilenceTimeoutSec: 5,
		},
	})

	require.NoError(t, s.mfsm.Handle(mediafsm.EvAnswer))

	go func() {
		time.Sleep(500 * time.Millisecond)

		// 第一轮对话。
		trans.audioCh <- loudFrame(160, 10000)
		time.Sleep(50 * time.Millisecond)

		streamCh1 := make(chan string, 1)
		dialogue.setStreamCh(streamCh1)
		asrStream.eventsCh <- aiface.ASREvent{Text: "第一轮", IsFinal: true, Confidence: 0.9}
		time.Sleep(100 * time.Millisecond)
		streamCh1 <- "第一轮回复"
		close(streamCh1)

		// 等播放完，再来第二轮。
		time.Sleep(500 * time.Millisecond)

		// 第二轮对话。
		trans.audioCh <- loudFrame(160, 10000)
		time.Sleep(50 * time.Millisecond)

		streamCh2 := make(chan string, 1)
		dialogue.setStreamCh(streamCh2)
		asrStream.eventsCh <- aiface.ASREvent{Text: "第二轮", IsFinal: true, Confidence: 0.95}
		time.Sleep(100 * time.Millisecond)
		streamCh2 <- "第二轮回复"
		close(streamCh2)

		time.Sleep(500 * time.Millisecond)
		close(trans.audioCh)
	}()

	result, err := s.Run(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	// 验证有多轮交互事件。
	eventTypes := make(map[EventType]int)
	for _, evt := range result.Events {
		eventTypes[evt.EventType]++
	}
	assert.GreaterOrEqual(t, eventTypes[EventBotSpeakStart], 2, "至少2次 bot speak")
}
