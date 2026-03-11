package sherpa

import (
	"errors"
	"log/slog"
	"os"
)

// testLogger 返回一个测试用的 logger。
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))
}

// --- fakeASREngine ---

type fakeASREngine struct {
	stream asrStream
	err    error
	closed bool
}

func (e *fakeASREngine) createStream() (asrStream, error) {
	if e.err != nil {
		return nil, e.err
	}
	return e.stream, nil
}

func (e *fakeASREngine) close() {
	e.closed = true
}

// --- fakeASRStream ---

type fakeASRStream struct {
	waveformAccepted bool
	readyCount       int // 每次 isReady 调用减 1，到 0 返回 false
	resultText       string
	endpoint         bool
	resetCalled      bool
	closeCalled      bool
	decodeCalled     int
}

func (s *fakeASRStream) acceptWaveform(_ int, _ []float32) {
	s.waveformAccepted = true
}

func (s *fakeASRStream) isReady() bool {
	if s.readyCount > 0 {
		s.readyCount--
		return true
	}
	return false
}

func (s *fakeASRStream) decode() {
	s.decodeCalled++
}

func (s *fakeASRStream) getResultText() string {
	return s.resultText
}

func (s *fakeASRStream) isEndpoint() bool {
	return s.endpoint
}

func (s *fakeASRStream) reset() {
	s.resetCalled = true
}

func (s *fakeASRStream) close() {
	s.closeCalled = true
}

// --- fakeVADEngine ---

type fakeVADEngine struct {
	speechResult bool
	acceptCount  int
	segments     [][]float32 // 模拟的 speech segments
	segIdx       int
	resetCalled  bool
	closed       bool
}

func (e *fakeVADEngine) acceptWaveform(_ []float32) {
	e.acceptCount++
}

func (e *fakeVADEngine) isSpeech() bool {
	return e.speechResult
}

func (e *fakeVADEngine) isEmpty() bool {
	return e.segIdx >= len(e.segments)
}

func (e *fakeVADEngine) frontSamples() []float32 {
	if e.segIdx >= len(e.segments) {
		return nil
	}
	return e.segments[e.segIdx]
}

func (e *fakeVADEngine) pop() {
	if e.segIdx < len(e.segments) {
		e.segIdx++
	}
}

func (e *fakeVADEngine) reset() {
	e.resetCalled = true
}

func (e *fakeVADEngine) close() {
	e.closed = true
}

// --- fakeTTSEngine ---

type fakeTTSEngine struct {
	samples        []float32
	fakeSampleRate int
	generateCalled bool
	closed         bool
}

func (e *fakeTTSEngine) generate(_ string, _ int, _ float32) []float32 {
	e.generateCalled = true
	return e.samples
}

func (e *fakeTTSEngine) sampleRate() int {
	return e.fakeSampleRate
}

func (e *fakeTTSEngine) close() {
	e.closed = true
}

// --- fakeEmbeddingEngine ---

type fakeEmbeddingEngine struct {
	embedding []float32
	err       error
	fakeDim   int
	closed    bool
}

func (e *fakeEmbeddingEngine) extract(_ int, _ []float32) ([]float32, error) {
	if e.err != nil {
		return nil, e.err
	}
	return e.embedding, nil
}

func (e *fakeEmbeddingEngine) dim() int {
	return e.fakeDim
}

func (e *fakeEmbeddingEngine) close() {
	e.closed = true
}

// --- fakeEmbeddingEngineFunc: 支持自定义 extract 行为 ---

type fakeEmbeddingEngineFunc struct {
	extractFunc func(sampleRate int, samples []float32) ([]float32, error)
	fakeDim     int
	closed      bool
}

func (e *fakeEmbeddingEngineFunc) extract(sampleRate int, samples []float32) ([]float32, error) {
	if e.extractFunc != nil {
		return e.extractFunc(sampleRate, samples)
	}
	return nil, errors.New("no extractFunc set")
}

func (e *fakeEmbeddingEngineFunc) dim() int {
	return e.fakeDim
}

func (e *fakeEmbeddingEngineFunc) close() {
	e.closed = true
}
