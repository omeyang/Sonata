package sherpa

//go:generate mockgen -source=backend.go -destination=mock_backend_test.go -package=sherpa

// providerCPU 是 sherpa-onnx 推理使用的 CPU provider 名称。
const providerCPU = "cpu"

// asrEngine abstracts sherpa-onnx OnlineRecognizer for testability.
type asrEngine interface {
	createStream() (asrStream, error)
	close()
}

// asrStream abstracts OnlineStream + recognizer decode operations.
type asrStream interface {
	acceptWaveform(sampleRate int, samples []float32)
	isReady() bool
	decode()
	getResultText() string
	isEndpoint() bool
	reset()
	close()
}

// vadEngine abstracts sherpa-onnx VoiceActivityDetector.
type vadEngine interface {
	acceptWaveform(samples []float32)
	isSpeech() bool
	isEmpty() bool
	frontSamples() []float32 // returns copy of front segment's samples, nil if empty
	pop()
	reset()
	close()
}

// ttsEngine abstracts sherpa-onnx OfflineTts.
type ttsEngine interface {
	generate(text string, speakerID int, speed float32) []float32 // returns float32 samples, nil on failure
	sampleRate() int
	close()
}

// embeddingEngine abstracts sherpa-onnx SpeakerEmbeddingExtractor.
type embeddingEngine interface {
	extract(sampleRate int, samples []float32) ([]float32, error)
	dim() int
	close()
}
