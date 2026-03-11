package sherpa

import (
	"errors"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// --- ASR ---

// sherpaASREngine wraps *sherpa.OnlineRecognizer.
type sherpaASREngine struct {
	recognizer *sherpa.OnlineRecognizer
}

func (e *sherpaASREngine) createStream() (asrStream, error) {
	s := sherpa.NewOnlineStream(e.recognizer)
	if s == nil {
		return nil, errors.New("sherpa: 创建 OnlineStream 失败")
	}
	return &sherpaASRStream{recognizer: e.recognizer, stream: s}, nil
}

func (e *sherpaASREngine) close() {
	sherpa.DeleteOnlineRecognizer(e.recognizer)
}

type sherpaASRStream struct {
	recognizer *sherpa.OnlineRecognizer
	stream     *sherpa.OnlineStream
}

func (s *sherpaASRStream) acceptWaveform(sampleRate int, samples []float32) {
	s.stream.AcceptWaveform(sampleRate, samples)
}

func (s *sherpaASRStream) isReady() bool {
	return s.recognizer.IsReady(s.stream)
}

func (s *sherpaASRStream) decode() {
	s.recognizer.Decode(s.stream)
}

func (s *sherpaASRStream) getResultText() string {
	return s.recognizer.GetResult(s.stream).Text
}

func (s *sherpaASRStream) isEndpoint() bool {
	return s.recognizer.IsEndpoint(s.stream)
}

func (s *sherpaASRStream) reset() {
	s.recognizer.Reset(s.stream)
}

func (s *sherpaASRStream) close() {
	sherpa.DeleteOnlineStream(s.stream)
}

// --- VAD ---

// sherpaVADEngine wraps *sherpa.VoiceActivityDetector.
type sherpaVADEngine struct {
	vad *sherpa.VoiceActivityDetector
}

func (e *sherpaVADEngine) acceptWaveform(samples []float32) {
	e.vad.AcceptWaveform(samples)
}

func (e *sherpaVADEngine) isSpeech() bool {
	return e.vad.IsSpeech()
}

func (e *sherpaVADEngine) isEmpty() bool {
	return e.vad.IsEmpty()
}

func (e *sherpaVADEngine) frontSamples() []float32 {
	seg := e.vad.Front()
	if seg == nil || len(seg.Samples) == 0 {
		return nil
	}
	cp := make([]float32, len(seg.Samples))
	copy(cp, seg.Samples)
	return cp
}

func (e *sherpaVADEngine) pop() {
	e.vad.Pop()
}

func (e *sherpaVADEngine) reset() {
	e.vad.Reset()
}

func (e *sherpaVADEngine) close() {
	sherpa.DeleteVoiceActivityDetector(e.vad)
}

// --- TTS ---

// sherpaTTSEngine wraps *sherpa.OfflineTts.
type sherpaTTSEngine struct {
	tts *sherpa.OfflineTts
}

func (e *sherpaTTSEngine) generate(text string, speakerID int, speed float32) []float32 {
	gen := e.tts.Generate(text, speakerID, speed)
	if gen == nil || len(gen.Samples) == 0 {
		return nil
	}
	return gen.Samples
}

func (e *sherpaTTSEngine) sampleRate() int {
	return e.tts.SampleRate()
}

func (e *sherpaTTSEngine) close() {
	sherpa.DeleteOfflineTts(e.tts)
}

// --- Embedding ---

// sherpaEmbeddingEngine wraps *sherpa.SpeakerEmbeddingExtractor.
type sherpaEmbeddingEngine struct {
	extractor *sherpa.SpeakerEmbeddingExtractor
}

func (e *sherpaEmbeddingEngine) extract(sampleRate int, samples []float32) ([]float32, error) {
	stream := e.extractor.CreateStream()
	if stream == nil {
		return nil, errors.New("sherpa: 创建声纹提取流失败")
	}
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)
	stream.InputFinished()

	if !e.extractor.IsReady(stream) {
		return nil, errors.New("sherpa: 音频过短，无法提取声纹")
	}

	embedding := e.extractor.Compute(stream)
	if len(embedding) == 0 {
		return nil, errors.New("sherpa: 声纹提取返回空向量")
	}

	result := make([]float32, len(embedding))
	copy(result, embedding)
	return result, nil
}

func (e *sherpaEmbeddingEngine) dim() int {
	return e.extractor.Dim()
}

func (e *sherpaEmbeddingEngine) close() {
	sherpa.DeleteSpeakerEmbeddingExtractor(e.extractor)
}
