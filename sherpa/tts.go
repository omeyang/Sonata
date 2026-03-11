package sherpa

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"unicode/utf8"

	"github.com/omeyang/Sonata/engine/pcm"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// LocalTTSConfig 配置本地离线 TTS（基于 sherpa-onnx OfflineTts）。
type LocalTTSConfig struct {
	// VITS 模型文件路径。
	ModelPath string

	// TokensPath tokens 文件路径。
	TokensPath string

	// DataDir 模型数据目录（如 espeak-ng-data）。
	DataDir string

	// DictDir 词典目录路径（可选）。
	DictDir string

	// LexiconPath 词典文件路径（可选）。
	LexiconPath string

	// RuleFsts 文本规范化 FST 路径（可选）。
	RuleFsts string

	// RuleFars 文本规范化 FAR 路径（可选）。
	RuleFars string

	// ModelType 模型类型，如 "vits"。留空由 sherpa-onnx 自动推断。
	ModelType string

	// NumThreads 推理线程数，默认 1。
	NumThreads int

	// SpeakerID 说话人 ID，多说话人模型中使用。默认 0。
	SpeakerID int

	// Speed 语速倍率，默认 1.0。
	Speed float32

	// SampleRate 输出采样率提示。
	// 如果为 0，使用模型自身的采样率。
	// 如果与模型采样率不同，合成后会自动重采样。
	SampleRate int

	// Logger 日志记录器，为 nil 时使用 slog.Default()。
	Logger *slog.Logger
}

func (c *LocalTTSConfig) setDefaults() {
	if c.NumThreads <= 0 {
		c.NumThreads = 1
	}
	if c.Speed <= 0 {
		c.Speed = 1.0
	}
}

// LocalTTS 封装 sherpa-onnx OfflineTts，提供本地文本转语音。
//
// 实现 aiface.TTSProvider 接口。
// 模型在创建时一次性加载，所有合成共享同一个 OfflineTts 实例。
// sherpa-onnx C 层非线程安全，合成操作通过互斥锁串行化。
type LocalTTS struct {
	engine     ttsEngine
	config     LocalTTSConfig
	logger     *slog.Logger
	sampleRate int // 模型实际采样率
	mu         sync.Mutex
}

// buildTTSConfig 从 LocalTTSConfig 构建 sherpa-onnx 配置。
// 纯 Go 逻辑，不涉及 C 调用。
func buildTTSConfig(cfg LocalTTSConfig) sherpa.OfflineTtsConfig {
	config := sherpa.OfflineTtsConfig{}
	config.Model.Vits.Model = cfg.ModelPath
	config.Model.Vits.Tokens = cfg.TokensPath
	config.Model.Vits.DataDir = cfg.DataDir
	config.Model.Vits.DictDir = cfg.DictDir
	config.Model.Vits.Lexicon = cfg.LexiconPath
	config.Model.NumThreads = cfg.NumThreads
	config.Model.Provider = providerCPU
	config.RuleFsts = cfg.RuleFsts
	config.RuleFars = cfg.RuleFars
	config.MaxNumSentences = 1
	return config
}

// NewLocalTTS 加载模型并创建本地 TTS 实例。
// 模型加载可能耗时数秒，建议在启动时调用。
// 调用方必须在使用完毕后调用 Close 释放 C 资源。
func NewLocalTTS(cfg LocalTTSConfig) (*LocalTTS, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("sherpa: LocalTTSConfig.ModelPath 不能为空")
	}

	cfg.setDefaults()

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	config := buildTTSConfig(cfg)

	logger.Info("加载本地 TTS 模型",
		slog.String("model", cfg.ModelPath),
	)

	tts := sherpa.NewOfflineTts(&config)
	if tts == nil {
		return nil, errors.New("sherpa: 创建 OfflineTts 失败")
	}

	eng := &sherpaTTSEngine{tts: tts}
	modelSampleRate := eng.sampleRate()

	logger.Info("本地 TTS 模型加载完成",
		slog.Int("sample_rate", modelSampleRate),
		slog.Int("num_speakers", tts.NumSpeakers()),
	)

	return &LocalTTS{
		engine:     eng,
		config:     cfg,
		logger:     logger,
		sampleRate: modelSampleRate,
	}, nil
}

// Synthesize 从完整文本生成 PCM16 LE 音频。
func (t *LocalTTS) Synthesize(_ context.Context, text string, _ any) ([]byte, error) {
	if text == "" {
		return nil, nil
	}

	t.mu.Lock()
	if t.engine == nil {
		t.mu.Unlock()
		return nil, errors.New("sherpa: LocalTTS 已关闭")
	}

	samples := t.engine.generate(text, t.config.SpeakerID, t.config.Speed)
	t.mu.Unlock()

	if samples == nil {
		return nil, nil
	}

	pcmData := pcm.Float32ToPCM16(samples)
	return pcmData, nil
}

// SynthesizeStream 接收文本通道并返回音频片段通道。
// 每段文本独立合成，合成完成后发送到音频通道。
func (t *LocalTTS) SynthesizeStream(ctx context.Context, textCh <-chan string, _ any) (<-chan []byte, error) {
	t.mu.Lock()
	if t.engine == nil {
		t.mu.Unlock()
		return nil, errors.New("sherpa: LocalTTS 已关闭")
	}
	t.mu.Unlock()

	audioCh := make(chan []byte, 8)

	go func() {
		defer close(audioCh)

		for {
			select {
			case <-ctx.Done():
				return
			case text, ok := <-textCh:
				if !ok {
					return
				}
				if text == "" {
					continue
				}

				pcmData, err := t.Synthesize(ctx, text, nil)
				if err != nil {
					t.logger.Warn("本地 TTS 合成失败",
						slog.String("error", err.Error()),
					)
					return
				}
				if pcmData == nil {
					continue
				}

				select {
				case audioCh <- pcmData:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return audioCh, nil
}

// Cancel 中止当前合成。
// 本地 TTS 的合成是同步阻塞的，Cancel 通过 context 取消实现。
func (t *LocalTTS) Cancel() error {
	return nil
}

// SampleRate 返回模型的采样率。
func (t *LocalTTS) SampleRate() int {
	return t.sampleRate
}

// Close 释放 OfflineTts 的 C 资源。
func (t *LocalTTS) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.engine == nil {
		return nil
	}

	t.engine.close()
	t.engine = nil
	return nil
}

// TextRuneCount 返回文本的 Unicode 字符数。
// 用于 TieredTTS 的路由决策。
func TextRuneCount(s string) int {
	return utf8.RuneCountInString(s)
}
