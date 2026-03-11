// Package sherpa 提供基于 sherpa-onnx 的本地推理适配器。
//
// 通过封装 sherpa-onnx 的 C API（via CGO），提供与 Sonata 核心接口兼容的
// 本地 VAD、ASR、TTS 实现，消除云端网络延迟。
//
// 本包作为独立 Go 模块（Sonata/sherpa），隔离 CGO 依赖，
// 不使用 sherpa-onnx 的消费者无需引入 CGO 工具链。
package sherpa

import (
	"errors"
	"fmt"
	"sync"

	"github.com/omeyang/Sonata/engine/pcm"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// SileroVADConfig 配置 Silero VAD 模型。
type SileroVADConfig struct {
	// ModelPath 是 silero_vad.onnx 模型文件路径。
	ModelPath string

	// Threshold 语音检测概率阈值。
	// 高于此值判定为语音，低于此值判定为静默。
	// 取值范围 (0, 1)，默认 0.5。电话场景建议 0.4-0.6。
	Threshold float32

	// MinSilenceDuration 最小静默时长（秒），用于判定一段语音结束。
	// 短于此值的静默会被合并到前后语音段中。
	// 默认 0.5。电话场景中可缩短到 0.3 以实现更快的端点检测。
	MinSilenceDuration float32

	// MinSpeechDuration 最小语音时长（秒），短于此值的语音会被丢弃。
	// 防止短暂噪音被误判为语音。默认 0.25。
	MinSpeechDuration float32

	// MaxSpeechDuration 最大单段语音时长（秒）。
	// 超过此值时强制分段，防止内存无限增长。默认 10。
	MaxSpeechDuration float32

	// SampleRate 音频采样率（Hz）。Silero VAD 期望 16000。
	SampleRate int
}

// sileroVADDefaults 填充零值字段为默认值。
func (c *SileroVADConfig) setDefaults() {
	if c.Threshold == 0 {
		c.Threshold = 0.5
	}
	if c.MinSilenceDuration == 0 {
		c.MinSilenceDuration = 0.5
	}
	if c.MinSpeechDuration == 0 {
		c.MinSpeechDuration = 0.25
	}
	if c.MaxSpeechDuration == 0 {
		c.MaxSpeechDuration = 10
	}
	if c.SampleRate == 0 {
		c.SampleRate = 16000
	}
}

// sileroWindowSize 是 Silero VAD 要求的固定窗口大小（采样数）。
// 16kHz 下 512 采样 = 32ms。
const sileroWindowSize = 512

// sileroBufferSeconds 是 VAD 内部环形缓冲区的大小（秒）。
const sileroBufferSeconds float32 = 5

// SileroVAD 封装 sherpa-onnx Silero VAD，提供帧级人声检测。
//
// 实现 session.SpeechDetector 接口（隐式满足）。
// sherpa-onnx C 层非线程安全，所有操作通过互斥锁串行化。
//
// 使用方式：
//
//	vad, err := sherpa.NewSileroVAD(cfg)
//	defer vad.Close()
//	isSpeech, _ := vad.IsSpeech(frame)
type SileroVAD struct {
	engine     vadEngine
	sampleRate int
	windowSize int
	buf        []float32 // 累积不足一个窗口的尾部采样
	mu         sync.Mutex
}

// buildVADConfig 从 SileroVADConfig 构建 sherpa-onnx 配置。
// 纯 Go 逻辑，不涉及 C 调用。
func buildVADConfig(cfg SileroVADConfig) sherpa.VadModelConfig {
	config := sherpa.VadModelConfig{}
	config.SileroVad.Model = cfg.ModelPath
	config.SileroVad.Threshold = cfg.Threshold
	config.SileroVad.MinSilenceDuration = cfg.MinSilenceDuration
	config.SileroVad.MinSpeechDuration = cfg.MinSpeechDuration
	config.SileroVad.MaxSpeechDuration = cfg.MaxSpeechDuration
	config.SileroVad.WindowSize = sileroWindowSize
	config.SampleRate = cfg.SampleRate
	config.NumThreads = 1
	config.Provider = providerCPU
	return config
}

// NewSileroVAD 创建 Silero VAD 实例。
// 调用方必须在使用完毕后调用 Close 释放 C 资源。
func NewSileroVAD(cfg SileroVADConfig) (*SileroVAD, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("sherpa: SileroVADConfig.ModelPath 不能为空")
	}

	cfg.setDefaults()

	config := buildVADConfig(cfg)

	vad := sherpa.NewVoiceActivityDetector(&config, sileroBufferSeconds)
	if vad == nil {
		return nil, errors.New("sherpa: 创建 Silero VAD 失败")
	}

	return &SileroVAD{
		engine:     &sherpaVADEngine{vad: vad},
		sampleRate: cfg.SampleRate,
		windowSize: sileroWindowSize,
	}, nil
}

// IsSpeech 检测音频帧是否包含人声。
//
// frame 为 PCM16 LE 单声道数据（与 Sonata audio 包的约定一致）。
// 输入采样率如果不是 16kHz，需要在调用前完成重采样。
//
// 内部将 PCM16 转换为 float32，并按 VAD 窗口大小（512 采样）分段处理。
// 不足一个窗口的尾部数据缓存到下一次调用。
func (v *SileroVAD) IsSpeech(frame []byte) (bool, error) {
	if len(frame) < 2 {
		return false, nil
	}

	samples := pcm.PCM16ToFloat32(frame)
	if samples == nil {
		return false, nil
	}

	v.mu.Lock()
	defer v.mu.Unlock()

	if v.engine == nil {
		return false, errors.New("sherpa: VAD 已关闭")
	}

	// 合并上次遗留的尾部数据。
	if len(v.buf) > 0 {
		samples = append(v.buf, samples...)
		v.buf = nil
	}

	// 按窗口大小逐段喂入 VAD。
	var speech bool
	for len(samples) >= v.windowSize {
		window := samples[:v.windowSize]
		samples = samples[v.windowSize:]
		v.engine.acceptWaveform(window)
		if v.engine.isSpeech() {
			speech = true
		}
	}

	// 缓存不足一个窗口的尾部。
	if len(samples) > 0 {
		v.buf = make([]float32, len(samples))
		copy(v.buf, samples)
	}

	return speech, nil
}

// SpeechSegments 返回 VAD 内部已分割完成的语音段。
// 每个段包含完整的 float32 采样数据。
// 返回后段从 VAD 内部队列中移除。
// 返回空切片表示没有已完成的语音段。
func (v *SileroVAD) SpeechSegments() [][]float32 {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.engine == nil {
		return nil
	}

	var segments [][]float32
	for !v.engine.isEmpty() {
		cp := v.engine.frontSamples()
		v.engine.pop()
		if cp != nil {
			segments = append(segments, cp)
		}
	}
	return segments
}

// Reset 清除 VAD 内部状态，准备处理新的音频流。
func (v *SileroVAD) Reset() {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.engine != nil {
		v.engine.reset()
	}
	v.buf = nil
}

// Close 释放底层 C 资源。
// Close 后调用 IsSpeech 等方法会返回错误。
// 重复调用 Close 是安全的。
func (v *SileroVAD) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.engine == nil {
		return nil
	}

	v.engine.close()
	v.engine = nil
	v.buf = nil
	return nil
}

// String 返回 VAD 的描述信息。
func (v *SileroVAD) String() string {
	return fmt.Sprintf("SileroVAD(sampleRate=%d, windowSize=%d)", v.sampleRate, v.windowSize)
}
