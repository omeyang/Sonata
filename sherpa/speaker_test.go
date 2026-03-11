package sherpa

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// === SpeakerEmbeddingConfig 测试 ===

func TestSpeakerEmbeddingConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := SpeakerEmbeddingConfig{ModelPath: "/fake/model.onnx"}
	cfg.setDefaults()

	assert.Equal(t, 1, cfg.NumThreads)
	assert.InDelta(t, 0.6, float64(cfg.Threshold), 0.001)
}

func TestSpeakerEmbeddingConfig_CustomValues(t *testing.T) {
	t.Parallel()

	cfg := SpeakerEmbeddingConfig{
		ModelPath:  "/model.onnx",
		NumThreads: 4,
		Threshold:  0.75,
	}
	cfg.setDefaults()

	assert.Equal(t, 4, cfg.NumThreads)
	assert.InDelta(t, 0.75, float64(cfg.Threshold), 0.001)
}

func TestNewSpeakerEmbedding_EmptyModelPath(t *testing.T) {
	t.Parallel()

	_, err := NewSpeakerEmbedding(SpeakerEmbeddingConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ModelPath")
}

// === SpeakerEmbedding Close 幂等性 ===

func TestSpeakerEmbedding_Close_Idempotent(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{engine: nil, dim: 192}

	assert.NoError(t, s.Close())
	assert.NoError(t, s.Close())
}

// === SpeakerEmbedding 关闭后操作 ===

func TestSpeakerEmbedding_Extract_AfterClose(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{engine: nil, dim: 192}

	_, err := s.Extract([]float32{0.1, 0.2, 0.3})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
}

func TestSpeakerEmbedding_Extract_EmptySamples(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{engine: nil, dim: 192}

	_, err := s.Extract(nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "为空")

	_, err = s.Extract([]float32{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "为空")
}

func TestSpeakerEmbedding_ExtractFromPCM16_InvalidData(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{engine: nil, dim: 192}

	// 单字节无法构成 PCM16 采样。
	_, err := s.ExtractFromPCM16([]byte{0x01})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "无效")
}

// === SpeakerEmbedding String ===

func TestSpeakerEmbedding_String(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{dim: 192}
	str := s.String()
	assert.Contains(t, str, "192")
	assert.Contains(t, str, "SpeakerEmbedding")
}

func TestSpeakerEmbedding_Dim(t *testing.T) {
	t.Parallel()

	s := &SpeakerEmbedding{dim: 256}
	assert.Equal(t, 256, s.Dim())
}

// === CosineSimilarity 测试 ===

func TestCosineSimilarity_IdenticalVectors(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2, 3, 4, 5}
	sim := CosineSimilarity(a, a)
	assert.InDelta(t, 1.0, float64(sim), 0.0001)
}

func TestCosineSimilarity_OppositeVectors(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2, 3}
	b := []float32{-1, -2, -3}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, -1.0, float64(sim), 0.0001)
}

func TestCosineSimilarity_OrthogonalVectors(t *testing.T) {
	t.Parallel()

	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, 0.0, float64(sim), 0.0001)
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	assert.Equal(t, float32(0), sim)
}

func TestCosineSimilarity_EmptyVectors(t *testing.T) {
	t.Parallel()

	assert.Equal(t, float32(0), CosineSimilarity(nil, nil))
	assert.Equal(t, float32(0), CosineSimilarity([]float32{}, []float32{}))
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	t.Parallel()

	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := CosineSimilarity(a, b)
	assert.Equal(t, float32(0), sim)
}

func TestCosineSimilarity_SimilarVectors(t *testing.T) {
	t.Parallel()

	// 两个近似但不完全相同的向量，相似度应接近 1。
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{1.1, 2.05, 2.95, 4.1, 4.9}
	sim := CosineSimilarity(a, b)
	assert.Greater(t, float64(sim), 0.99)
}

func TestCosineSimilarity_ScaledVectors(t *testing.T) {
	t.Parallel()

	// 余弦相似度不受向量缩放影响。
	a := []float32{1, 2, 3}
	b := []float32{10, 20, 30}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, 1.0, float64(sim), 0.0001)
}

func TestCosineSimilarity_NumericalStability(t *testing.T) {
	t.Parallel()

	// 使用很小的值验证数值稳定性。
	a := []float32{1e-10, 2e-10, 3e-10}
	b := []float32{1e-10, 2e-10, 3e-10}
	sim := CosineSimilarity(a, b)
	assert.InDelta(t, 1.0, float64(sim), 0.001)
	assert.False(t, math.IsNaN(float64(sim)))
}

// === SpeakerTrackerConfig 测试 ===

func TestSpeakerTrackerConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := SpeakerTrackerConfig{}
	cfg.setDefaults()

	assert.InDelta(t, 0.6, float64(cfg.Threshold), 0.001)
	assert.Equal(t, 3, cfg.SwitchCount)
}

func TestSpeakerTrackerConfig_CustomValues(t *testing.T) {
	t.Parallel()

	cfg := SpeakerTrackerConfig{
		Threshold:   0.5,
		SwitchCount: 5,
	}
	cfg.setDefaults()

	assert.InDelta(t, 0.5, float64(cfg.Threshold), 0.001)
	assert.Equal(t, 5, cfg.SwitchCount)
}

func TestNewSpeakerTracker_NilExtractor(t *testing.T) {
	t.Parallel()

	_, err := NewSpeakerTracker(SpeakerTrackerConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Extractor")
}

// === SpeakerTracker 状态管理测试 ===

func TestSpeakerTracker_Reset(t *testing.T) {
	t.Parallel()

	// 直接构造 tracker 以测试 Reset 逻辑。
	tracker := &SpeakerTracker{
		primaryEmb:      []float32{1, 2, 3},
		threshold:       0.6,
		switchThreshold: 3,
		consecutiveMiss: 2,
		switched:        true,
	}

	tracker.Reset()

	assert.Nil(t, tracker.PrimaryEmbedding())
	assert.False(t, tracker.HasSwitched())
}

func TestSpeakerTracker_PrimaryEmbedding_Nil(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		threshold:       0.6,
		switchThreshold: 3,
	}

	assert.Nil(t, tracker.PrimaryEmbedding())
}

func TestSpeakerTracker_PrimaryEmbedding_Copy(t *testing.T) {
	t.Parallel()

	original := []float32{1, 2, 3, 4, 5}
	tracker := &SpeakerTracker{
		primaryEmb:      original,
		threshold:       0.6,
		switchThreshold: 3,
	}

	emb := tracker.PrimaryEmbedding()
	require.NotNil(t, emb)
	assert.Equal(t, original, emb)

	// 修改返回的副本不影响内部状态。
	emb[0] = 999
	assert.InDelta(t, 1.0, float64(tracker.primaryEmb[0]), 0.001)
}

func TestSpeakerTracker_HasSwitched_Initial(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		threshold:       0.6,
		switchThreshold: 3,
	}

	assert.False(t, tracker.HasSwitched())
}

// === SpeakerTracker Track 逻辑测试（使用 mock engine） ===

func TestSpeakerTracker_Track_FirstCallRegistersPrimary(t *testing.T) {
	t.Parallel()

	mock := &fakeEmbeddingEngine{
		embedding: []float32{0.5, 0.3, 0.8},
		fakeDim:   3,
	}
	extractor := &SpeakerEmbedding{engine: mock, dim: 3}

	tracker, err := NewSpeakerTracker(SpeakerTrackerConfig{
		Extractor: extractor,
		Threshold: 0.6,
	})
	require.NoError(t, err)

	switched, err := tracker.Track([]float32{0.1, 0.2})
	require.NoError(t, err)
	assert.False(t, switched)
	assert.NotNil(t, tracker.PrimaryEmbedding())
}

func TestSpeakerTracker_Track_MatchResetsCount(t *testing.T) {
	t.Parallel()

	// 第一次调用返回主声纹，后续返回相同声纹（高相似度）。
	primaryEmb := []float32{1, 0, 0}
	mock := &fakeEmbeddingEngine{
		embedding: primaryEmb,
		fakeDim:   3,
	}
	extractor := &SpeakerEmbedding{engine: mock, dim: 3}

	tracker, err := NewSpeakerTracker(SpeakerTrackerConfig{
		Extractor: extractor,
		Threshold: 0.6,
	})
	require.NoError(t, err)

	// 注册主声纹。
	_, err = tracker.Track([]float32{0.1})
	require.NoError(t, err)

	// 手动设置偏离计数。
	tracker.consecutiveMiss = 2

	// 匹配声纹应重置偏离计数。
	switched, err := tracker.Track([]float32{0.1})
	require.NoError(t, err)
	assert.False(t, switched)
	assert.Equal(t, 0, tracker.consecutiveMiss)
}

func TestSpeakerTracker_Track_SwitchDetection(t *testing.T) {
	t.Parallel()

	callCount := 0
	primaryEmb := []float32{1, 0, 0}
	mismatchEmb := []float32{0, 1, 0} // 正交，余弦相似度 = 0

	mock := &fakeEmbeddingEngineFunc{
		extractFunc: func(_ int, _ []float32) ([]float32, error) {
			callCount++
			if callCount == 1 {
				return primaryEmb, nil
			}
			return mismatchEmb, nil
		},
		fakeDim: 3,
	}
	extractor := &SpeakerEmbedding{engine: mock, dim: 3}

	tracker, err := NewSpeakerTracker(SpeakerTrackerConfig{
		Extractor:   extractor,
		Threshold:   0.6,
		SwitchCount: 3,
	})
	require.NoError(t, err)

	// 第 1 次：注册主声纹。
	switched, err := tracker.Track([]float32{0.1})
	require.NoError(t, err)
	assert.False(t, switched)

	// 第 2-4 次：偏离。
	for i := range 2 {
		switched, err = tracker.Track([]float32{0.1})
		require.NoError(t, err)
		assert.False(t, switched, "call %d should not switch", i+2)
	}

	// 第 5 次：达到阈值，触发切换。
	switched, err = tracker.Track([]float32{0.1})
	require.NoError(t, err)
	assert.True(t, switched)
	assert.True(t, tracker.HasSwitched())
}

func TestSpeakerTracker_Track_SwitchOnlyOnce(t *testing.T) {
	t.Parallel()

	callCount := 0
	primaryEmb := []float32{1, 0, 0}
	mismatchEmb := []float32{0, 1, 0}

	mock := &fakeEmbeddingEngineFunc{
		extractFunc: func(_ int, _ []float32) ([]float32, error) {
			callCount++
			if callCount == 1 {
				return primaryEmb, nil
			}
			return mismatchEmb, nil
		},
		fakeDim: 3,
	}
	extractor := &SpeakerEmbedding{engine: mock, dim: 3}

	tracker, err := NewSpeakerTracker(SpeakerTrackerConfig{
		Extractor:   extractor,
		Threshold:   0.6,
		SwitchCount: 2,
	})
	require.NoError(t, err)

	// 注册主声纹。
	_, _ = tracker.Track([]float32{0.1})

	// 偏离 2 次触发切换。
	_, _ = tracker.Track([]float32{0.1})
	switched, _ := tracker.Track([]float32{0.1})
	assert.True(t, switched)

	// 继续偏离不再触发。
	switched, _ = tracker.Track([]float32{0.1})
	assert.False(t, switched)
	assert.True(t, tracker.HasSwitched())
}

func TestSpeakerTracker_Reset_AllowsRetrigger(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		primaryEmb:      []float32{1, 0, 0},
		threshold:       0.6,
		switchThreshold: 3,
		consecutiveMiss: 5,
		switched:        true,
	}

	tracker.Reset()

	assert.False(t, tracker.HasSwitched())
	assert.Nil(t, tracker.PrimaryEmbedding())
	assert.Equal(t, 0, tracker.consecutiveMiss)
}

func TestSpeakerTracker_Track_ExtractError(t *testing.T) {
	t.Parallel()

	mock := &fakeEmbeddingEngine{
		err:     assert.AnError,
		fakeDim: 3,
	}
	extractor := &SpeakerEmbedding{engine: mock, dim: 3}

	tracker, err := NewSpeakerTracker(SpeakerTrackerConfig{
		Extractor: extractor,
	})
	require.NoError(t, err)

	_, err = tracker.Track([]float32{0.1, 0.2})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "声纹提取失败")
}

// === buildEmbeddingConfig ===

func TestBuildEmbeddingConfig(t *testing.T) {
	t.Parallel()

	cfg := SpeakerEmbeddingConfig{
		ModelPath:  "/model.onnx",
		NumThreads: 4,
		Threshold:  0.7,
	}
	cfg.setDefaults()

	config := buildEmbeddingConfig(cfg)

	assert.Equal(t, "/model.onnx", config.Model)
	assert.Equal(t, 4, config.NumThreads)
	assert.Equal(t, "cpu", config.Provider)
}

// === Mock-based SpeakerEmbedding tests ===

func TestSpeakerEmbedding_Extract_WithMock_Success(t *testing.T) {
	t.Parallel()

	expectedEmb := []float32{0.1, 0.2, 0.3, 0.4}
	mock := &fakeEmbeddingEngine{
		embedding: expectedEmb,
		fakeDim:   4,
	}
	s := &SpeakerEmbedding{engine: mock, dim: 4}

	result, err := s.Extract([]float32{0.5, 0.6})
	require.NoError(t, err)
	assert.Equal(t, expectedEmb, result)
}

func TestSpeakerEmbedding_Extract_WithMock_Error(t *testing.T) {
	t.Parallel()

	mock := &fakeEmbeddingEngine{
		err:     assert.AnError,
		fakeDim: 4,
	}
	s := &SpeakerEmbedding{engine: mock, dim: 4}

	_, err := s.Extract([]float32{0.5, 0.6})
	assert.Error(t, err)
}

func TestSpeakerEmbedding_ExtractFromPCM16_WithMock_Success(t *testing.T) {
	t.Parallel()

	expectedEmb := []float32{0.1, 0.2, 0.3}
	mock := &fakeEmbeddingEngine{
		embedding: expectedEmb,
		fakeDim:   3,
	}
	s := &SpeakerEmbedding{engine: mock, dim: 3}

	// 4 字节 = 2 个 PCM16 采样。
	pcmData := makePCM16Frame(2, 1000)
	result, err := s.ExtractFromPCM16(pcmData)
	require.NoError(t, err)
	assert.Equal(t, expectedEmb, result)
}

func TestSpeakerEmbedding_Close_WithMock(t *testing.T) {
	t.Parallel()

	mock := &fakeEmbeddingEngine{fakeDim: 4}
	s := &SpeakerEmbedding{engine: mock, dim: 4}

	assert.NoError(t, s.Close())
	assert.True(t, mock.closed)
	assert.Nil(t, s.engine)
}

// === 旧的内部状态测试（保留兼容） ===

func TestSpeakerTracker_TrackLogic_FirstCall_RegistersPrimary(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		threshold:       0.6,
		switchThreshold: 3,
	}

	// 模拟设置 primary embedding。
	tracker.mu.Lock()
	tracker.primaryEmb = []float32{0.5, 0.3, 0.8}
	tracker.mu.Unlock()

	assert.NotNil(t, tracker.PrimaryEmbedding())
	assert.False(t, tracker.HasSwitched())
}

func TestSpeakerTracker_TrackLogic_SwitchDetection(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		primaryEmb:      []float32{1, 0, 0},
		threshold:       0.6,
		switchThreshold: 3,
		consecutiveMiss: 2,
	}

	tracker.mu.Lock()
	tracker.consecutiveMiss++
	if tracker.consecutiveMiss >= tracker.switchThreshold && !tracker.switched {
		tracker.switched = true
	}
	tracker.mu.Unlock()

	assert.True(t, tracker.HasSwitched())
}

func TestSpeakerTracker_TrackLogic_MatchResetsCount(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		primaryEmb:      []float32{1, 0, 0},
		threshold:       0.6,
		switchThreshold: 3,
		consecutiveMiss: 2,
	}

	tracker.mu.Lock()
	tracker.consecutiveMiss = 0
	tracker.mu.Unlock()

	assert.Equal(t, 0, tracker.consecutiveMiss)
	assert.False(t, tracker.HasSwitched())
}

func TestSpeakerTracker_TrackLogic_SwitchOnlyOnce(t *testing.T) {
	t.Parallel()

	tracker := &SpeakerTracker{
		primaryEmb:      []float32{1, 0, 0},
		threshold:       0.6,
		switchThreshold: 3,
		consecutiveMiss: 3,
		switched:        true,
	}

	tracker.mu.Lock()
	tracker.consecutiveMiss++
	triggered := tracker.consecutiveMiss >= tracker.switchThreshold && !tracker.switched
	tracker.mu.Unlock()

	assert.False(t, triggered)
	assert.True(t, tracker.HasSwitched())
}
