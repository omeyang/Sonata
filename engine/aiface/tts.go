package aiface

import "context"

// TTSConfig 持有 TTS 调用的配置。
type TTSConfig struct {
	Model      string
	Voice      string
	SampleRate int
}

// TTSProvider 将文本合成为语音。
type TTSProvider interface {
	// SynthesizeStream 接收文本通道并返回音频片段通道。
	// 合成完成后音频通道关闭。
	SynthesizeStream(ctx context.Context, textCh <-chan string, cfg TTSConfig) (<-chan []byte, error)

	// Synthesize 从完整文本生成音频（用于预合成）。
	Synthesize(ctx context.Context, text string, cfg TTSConfig) ([]byte, error)

	// Cancel 中止当前合成（打断场景）。
	Cancel() error
}
