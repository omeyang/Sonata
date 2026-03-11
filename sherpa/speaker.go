package sherpa

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/omeyang/Sonata/engine/pcm"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// SpeakerEmbeddingConfig 声纹提取配置。
type SpeakerEmbeddingConfig struct {
	// ModelPath 是 3D-Speaker 或 WeSpeaker ONNX 模型文件路径。
	ModelPath string

	// NumThreads 推理线程数，默认 1。
	NumThreads int

	// Threshold 相似度阈值，低于此值判定为不同人。
	// 取值范围 (0, 1)，默认 0.6。
	Threshold float32
}

func (c *SpeakerEmbeddingConfig) setDefaults() {
	if c.NumThreads <= 0 {
		c.NumThreads = 1
	}
	if c.Threshold <= 0 {
		c.Threshold = 0.6
	}
}

// SpeakerEmbedding 本地声纹提取器。
//
// 封装 sherpa-onnx SpeakerEmbeddingExtractor，
// 从音频片段中提取固定维度的声纹向量（embedding），
// 可用于余弦相似度比对判断是否同一说话人。
//
// sherpa-onnx C 层非线程安全，所有操作通过互斥锁串行化。
type SpeakerEmbedding struct {
	engine embeddingEngine
	dim    int
	mu     sync.Mutex
}

// buildEmbeddingConfig 从 SpeakerEmbeddingConfig 构建 sherpa-onnx 配置。
// 纯 Go 逻辑，不涉及 C 调用。
func buildEmbeddingConfig(cfg SpeakerEmbeddingConfig) *sherpa.SpeakerEmbeddingExtractorConfig {
	return &sherpa.SpeakerEmbeddingExtractorConfig{
		Model:      cfg.ModelPath,
		NumThreads: cfg.NumThreads,
		Provider:   providerCPU,
	}
}

// NewSpeakerEmbedding 加载模型并创建声纹提取器实例。
// 调用方必须在使用完毕后调用 Close 释放 C 资源。
func NewSpeakerEmbedding(cfg SpeakerEmbeddingConfig) (*SpeakerEmbedding, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("sherpa: SpeakerEmbeddingConfig.ModelPath 不能为空")
	}

	cfg.setDefaults()

	config := buildEmbeddingConfig(cfg)

	extractor := sherpa.NewSpeakerEmbeddingExtractor(config)
	if extractor == nil {
		return nil, errors.New("sherpa: 创建 SpeakerEmbeddingExtractor 失败")
	}

	eng := &sherpaEmbeddingEngine{extractor: extractor}
	dim := eng.dim()
	if dim <= 0 {
		eng.close()
		return nil, fmt.Errorf("sherpa: 声纹 embedding 维度异常: %d", dim)
	}

	return &SpeakerEmbedding{
		engine: eng,
		dim:    dim,
	}, nil
}

// Dim 返回 embedding 向量维度。
func (s *SpeakerEmbedding) Dim() int {
	return s.dim
}

// Extract 从一段音频中提取声纹 embedding 向量。
//
// samples 为 16kHz 单声道 float32 采样数据。
// 音频时长建议 >= 0.5 秒以获得可靠结果。
// 返回固定维度的 float32 向量。
func (s *SpeakerEmbedding) Extract(samples []float32) ([]float32, error) {
	if len(samples) == 0 {
		return nil, errors.New("sherpa: 音频采样为空")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.engine == nil {
		return nil, errors.New("sherpa: SpeakerEmbedding 已关闭")
	}

	return s.engine.extract(16000, samples)
}

// ExtractFromPCM16 从 PCM16 LE 音频数据中提取声纹 embedding。
// 内部将 PCM16 转换为 float32 后调用 Extract。
func (s *SpeakerEmbedding) ExtractFromPCM16(pcmData []byte) ([]float32, error) {
	samples := pcm.PCM16ToFloat32(pcmData)
	if samples == nil {
		return nil, errors.New("sherpa: PCM16 音频数据无效")
	}
	return s.Extract(samples)
}

// Close 释放底层 C 资源。
// 重复调用 Close 是安全的。
func (s *SpeakerEmbedding) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.engine == nil {
		return nil
	}

	s.engine.close()
	s.engine = nil
	return nil
}

// String 返回提取器的描述信息。
func (s *SpeakerEmbedding) String() string {
	return fmt.Sprintf("SpeakerEmbedding(dim=%d)", s.dim)
}

// CosineSimilarity 计算两个 embedding 的余弦相似度。
//
// 返回值范围 [-1, 1]，1 表示完全相同，0 表示无关，-1 表示完全相反。
// 说话人验证典型阈值：0.5-0.7。
//
// 如果任一向量为零向量，返回 0。
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}

	return float32(dot / denom)
}

// defaultSwitchThreshold 默认触发说话人切换所需的连续偏离次数。
const defaultSwitchThreshold = 3

// SpeakerTrackerConfig 说话人变更跟踪配置。
type SpeakerTrackerConfig struct {
	// Extractor 声纹提取器实例。
	Extractor *SpeakerEmbedding

	// Threshold 余弦相似度阈值，低于此值判定为不同说话人。
	// 默认使用 Extractor 对应的 SpeakerEmbeddingConfig.Threshold。
	// 如果为 0，使用默认值 0.6。
	Threshold float32

	// SwitchCount 触发说话人切换事件所需的连续偏离次数。
	// 防止单次噪音导致误判。默认 3。
	SwitchCount int
}

func (c *SpeakerTrackerConfig) setDefaults() {
	if c.Threshold <= 0 {
		c.Threshold = 0.6
	}
	if c.SwitchCount <= 0 {
		c.SwitchCount = defaultSwitchThreshold
	}
}

// SpeakerTracker 在会话中跟踪说话人切换。
//
// 使用流程：
//  1. 通话接通后，前几段音频通过 Track 自动注册为主说话人声纹。
//  2. 后续每段 ASR final 对应的音频调用 Track 比对。
//  3. 连续多段（SwitchCount 次）相似度低于阈值，返回 switched=true。
//  4. 对话引擎收到 switched 后可切换到身份确认流程。
type SpeakerTracker struct {
	extractor       *SpeakerEmbedding
	primaryEmb      []float32 // 主说话人声纹
	threshold       float32   // 相似度阈值
	switchThreshold int       // 触发切换所需连续次数
	consecutiveMiss int       // 当前连续偏离次数
	switched        bool      // 是否已触发切换
	mu              sync.Mutex
}

// NewSpeakerTracker 创建说话人变更跟踪器。
func NewSpeakerTracker(cfg SpeakerTrackerConfig) (*SpeakerTracker, error) {
	if cfg.Extractor == nil {
		return nil, errors.New("sherpa: SpeakerTrackerConfig.Extractor 不能为 nil")
	}

	cfg.setDefaults()

	return &SpeakerTracker{
		extractor:       cfg.Extractor,
		threshold:       cfg.Threshold,
		switchThreshold: cfg.SwitchCount,
	}, nil
}

// Track 传入一段音频采样，返回是否发生说话人切换。
//
// 首次调用时注册主说话人声纹，后续调用比对。
// switched 只在连续偏离次数达到 SwitchCount 阈值的那一刻返回 true，
// 之后持续偏离不再重复触发。
func (t *SpeakerTracker) Track(samples []float32) (switched bool, err error) {
	embedding, err := t.extractor.Extract(samples)
	if err != nil {
		return false, fmt.Errorf("声纹提取失败: %w", err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// 首次调用，注册主说话人声纹。
	if t.primaryEmb == nil {
		t.primaryEmb = embedding
		return false, nil
	}

	similarity := CosineSimilarity(t.primaryEmb, embedding)

	if similarity >= t.threshold {
		// 与主说话人匹配，重置连续偏离计数。
		t.consecutiveMiss = 0
		return false, nil
	}

	// 相似度低于阈值，累加偏离计数。
	t.consecutiveMiss++

	if t.consecutiveMiss >= t.switchThreshold && !t.switched {
		t.switched = true
		return true, nil
	}

	return false, nil
}

// Reset 重置跟踪状态，清除主说话人声纹。
// 下次 Track 调用将重新注册主说话人。
func (t *SpeakerTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.primaryEmb = nil
	t.consecutiveMiss = 0
	t.switched = false
}

// PrimaryEmbedding 返回当前主说话人声纹的副本。
// 如果尚未注册主说话人，返回 nil。
func (t *SpeakerTracker) PrimaryEmbedding() []float32 {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.primaryEmb == nil {
		return nil
	}

	cp := make([]float32, len(t.primaryEmb))
	copy(cp, t.primaryEmb)
	return cp
}

// HasSwitched 返回是否已检测到说话人切换。
func (t *SpeakerTracker) HasSwitched() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.switched
}
