package sherpa

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/omeyang/Sonata/engine/aiface"
	"github.com/omeyang/Sonata/engine/pcm"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// LocalASRConfig 配置本地流式 ASR（基于 sherpa-onnx OnlineRecognizer）。
type LocalASRConfig struct {
	// Paraformer 模型文件路径。
	EncoderPath string
	DecoderPath string
	TokensPath  string

	// ModelType 模型类型提示，加快模型加载。
	// 留空时由 sherpa-onnx 自动推断。
	ModelType string

	// NumThreads 推理线程数，默认 1。
	NumThreads int

	// SampleRate 期望的输入采样率，默认 16000。
	SampleRate int

	// EnableEndpoint 是否启用端点检测。默认启用。
	EnableEndpoint bool

	// Rule1MinTrailingSilence 长话语后的最小尾部静默（秒）。
	// 当已识别文本较长时，超过此静默时长即触发端点。默认 2.4。
	Rule1MinTrailingSilence float32

	// Rule2MinTrailingSilence 短话语后的最小尾部静默（秒）。
	// 当已识别文本较短时，超过此静默时长即触发端点。默认 1.2。
	// 电话场景建议 0.8 以加速端点检测。
	Rule2MinTrailingSilence float32

	// Rule3MinUtteranceLength 触发强制端点的最小话语长度（秒）。默认 20。
	Rule3MinUtteranceLength float32

	// Logger 日志记录器，为 nil 时使用 slog.Default()。
	Logger *slog.Logger
}

func (c *LocalASRConfig) setDefaults() {
	if c.NumThreads <= 0 {
		c.NumThreads = 1
	}
	if c.SampleRate <= 0 {
		c.SampleRate = 16000
	}
	if c.Rule1MinTrailingSilence == 0 {
		c.Rule1MinTrailingSilence = 2.4
	}
	if c.Rule2MinTrailingSilence == 0 {
		c.Rule2MinTrailingSilence = 1.2
	}
	if c.Rule3MinUtteranceLength == 0 {
		c.Rule3MinUtteranceLength = 20
	}
}

// LocalASR 封装 sherpa-onnx OnlineRecognizer，提供本地流式语音识别。
//
// 实现 aiface.ASRProvider 接口。
// 模型在创建时一次性加载，所有流共享同一个 Recognizer 实例。
// 每个流（localASRStream）维护独立的 OnlineStream 和解码 goroutine。
type LocalASR struct {
	engine asrEngine
	config LocalASRConfig
	logger *slog.Logger
	mu     sync.Mutex // 保护 engine 的并发访问
}

// buildASRConfig 从 LocalASRConfig 构建 sherpa-onnx 配置。
// 纯 Go 逻辑，不涉及 C 调用。
func buildASRConfig(cfg LocalASRConfig) sherpa.OnlineRecognizerConfig {
	config := sherpa.OnlineRecognizerConfig{}
	config.FeatConfig = sherpa.FeatureConfig{SampleRate: cfg.SampleRate, FeatureDim: 80}
	config.ModelConfig.Paraformer.Encoder = cfg.EncoderPath
	config.ModelConfig.Paraformer.Decoder = cfg.DecoderPath
	config.ModelConfig.Tokens = cfg.TokensPath
	config.ModelConfig.NumThreads = cfg.NumThreads
	config.ModelConfig.Provider = providerCPU
	config.ModelConfig.ModelType = cfg.ModelType
	config.DecodingMethod = "greedy_search"

	if cfg.EnableEndpoint {
		config.EnableEndpoint = 1
	}
	config.Rule1MinTrailingSilence = cfg.Rule1MinTrailingSilence
	config.Rule2MinTrailingSilence = cfg.Rule2MinTrailingSilence
	config.Rule3MinUtteranceLength = cfg.Rule3MinUtteranceLength

	return config
}

// NewLocalASR 加载模型并创建本地 ASR 实例。
// 模型加载可能耗时数秒，建议在启动时调用。
// 调用方必须在使用完毕后调用 Close 释放 C 资源。
func NewLocalASR(cfg LocalASRConfig) (*LocalASR, error) {
	if cfg.TokensPath == "" {
		return nil, errors.New("sherpa: LocalASRConfig.TokensPath 不能为空")
	}

	cfg.setDefaults()

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	config := buildASRConfig(cfg)

	logger.Info("加载本地 ASR 模型",
		slog.String("tokens", cfg.TokensPath),
		slog.String("encoder", cfg.EncoderPath),
	)

	recognizer := sherpa.NewOnlineRecognizer(&config)
	if recognizer == nil {
		return nil, errors.New("sherpa: 创建 OnlineRecognizer 失败")
	}

	logger.Info("本地 ASR 模型加载完成")

	return &LocalASR{
		engine: &sherpaASREngine{recognizer: recognizer},
		config: cfg,
		logger: logger,
	}, nil
}

// StartStream 打开一个新的本地识别流。
// 每个流独立运行解码循环，通过 Events() 通道输出识别结果。
func (a *LocalASR) StartStream(_ context.Context, _ aiface.ASRConfig) (aiface.ASRStream, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.engine == nil {
		return nil, errors.New("sherpa: LocalASR 已关闭")
	}

	stream, err := a.engine.createStream()
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())
	s := &localASRStream{
		stream:     stream,
		events:     make(chan aiface.ASREvent, 32),
		sampleRate: a.config.SampleRate,
		logger:     a.logger,
		cancel:     cancel,
		parentMu:   &a.mu,
	}

	go s.decodeLoop(ctx)

	return s, nil
}

// Close 释放 OnlineRecognizer 的 C 资源。
// 所有正在运行的流应在 Close 之前关闭。
func (a *LocalASR) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.engine == nil {
		return nil
	}

	a.engine.close()
	a.engine = nil
	return nil
}

// decodeInterval 解码循环的轮询间隔。
// 100ms 在延迟和 CPU 使用率间取得平衡。
const decodeInterval = 100 * time.Millisecond

// localASRStream 是 LocalASR 创建的流式识别会话。
// 实现 aiface.ASRStream 接口。
type localASRStream struct {
	stream     asrStream
	events     chan aiface.ASREvent
	sampleRate int
	logger     *slog.Logger
	cancel     context.CancelFunc
	parentMu   *sync.Mutex // 共享 engine 级互斥锁

	mu       sync.Mutex
	closed   bool
	lastText string    // 上次 partial 文本，避免重复发送
	startAt  time.Time // 流创建时间，用于计算延迟
}

// FeedAudio 向识别流发送音频片段。
// chunk 为 PCM16 LE 数据，内部转换为 float32 后喂入 sherpa-onnx。
func (s *localASRStream) FeedAudio(_ context.Context, chunk []byte) error {
	samples := pcm.PCM16ToFloat32(chunk)
	if samples == nil {
		return nil
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return errors.New("sherpa: ASR 流已关闭")
	}
	if s.startAt.IsZero() {
		s.startAt = time.Now()
	}
	s.mu.Unlock()

	s.parentMu.Lock()
	s.stream.acceptWaveform(s.sampleRate, samples)
	s.parentMu.Unlock()

	return nil
}

// Events 返回接收 ASR 事件的通道。
func (s *localASRStream) Events() <-chan aiface.ASREvent {
	return s.events
}

// Close 终止识别流并释放资源。
func (s *localASRStream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()

	s.cancel()

	s.parentMu.Lock()
	if s.stream != nil {
		s.stream.close()
		s.stream = nil
	}
	s.parentMu.Unlock()

	return nil
}

// decodeLoop 持续解码并通过 events 通道输出识别结果。
// partial 结果在文本变化时发送，endpoint 检测到时发送 final 结果。
func (s *localASRStream) decodeLoop(ctx context.Context) {
	defer close(s.events)

	ticker := time.NewTicker(decodeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.decodeTick(ctx)
		}
	}
}

// decodeTick 执行一次解码并发送结果。
func (s *localASRStream) decodeTick(ctx context.Context) {
	s.parentMu.Lock()

	// 解码所有就绪帧。
	for s.stream.isReady() {
		s.stream.decode()
	}

	text := s.stream.getResultText()
	isEndpoint := s.stream.isEndpoint()

	if isEndpoint && text != "" {
		s.stream.reset()
	}

	s.parentMu.Unlock()

	if text == "" {
		if isEndpoint {
			// 空文本的 endpoint（静默结束），重置但不发送事件。
			s.parentMu.Lock()
			s.stream.reset()
			s.parentMu.Unlock()
		}
		return
	}

	s.mu.Lock()
	latencyMs := int(time.Since(s.startAt).Milliseconds())
	lastText := s.lastText
	s.mu.Unlock()

	if isEndpoint {
		// 端点检测到：发送 final 结果。
		evt := aiface.ASREvent{
			Text:       text,
			IsFinal:    true,
			Confidence: 1.0, // 本地 ASR 不提供置信度，固定为 1.0。
			LatencyMs:  latencyMs,
		}
		select {
		case s.events <- evt:
		case <-ctx.Done():
			return
		}

		s.mu.Lock()
		s.lastText = ""
		s.mu.Unlock()
	} else if text != lastText {
		// 文本变化：发送 partial 结果。
		evt := aiface.ASREvent{
			Text:       text,
			IsFinal:    false,
			Confidence: 0.0,
			LatencyMs:  latencyMs,
		}
		select {
		case s.events <- evt:
		default:
			// 通道满时丢弃 partial（不阻塞解码循环）。
		}

		s.mu.Lock()
		s.lastText = text
		s.mu.Unlock()
	}
}
