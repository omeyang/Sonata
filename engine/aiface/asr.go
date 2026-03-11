// Package aiface 定义 AI 能力（ASR、LLM、TTS）和对话引擎的流式接口。
//
// 所有接口以流式优先设计，实现由各产品自行提供。
// 接口设计遵循 Go 惯例：小接口、行为命名、使用方定义。
package aiface

import "context"

// ASREvent 表示来自 ASR 流的识别事件。
type ASREvent struct {
	Text       string  // 识别出的文本。
	IsFinal    bool    // 是否为最终（非部分）结果。
	Confidence float64 // 识别置信度（0.0–1.0）。
	LatencyMs  int     // 从音频开始的延迟（毫秒）。
}

// ASRConfig 持有 ASR 流的配置。
type ASRConfig struct {
	Model      string
	SampleRate int
	Language   string
}

// ASRStream 是一个活跃的语音识别会话。
type ASRStream interface {
	// FeedAudio 向识别流发送音频片段。
	FeedAudio(ctx context.Context, chunk []byte) error

	// Events 返回接收 ASR 事件的通道。
	Events() <-chan ASREvent

	// Close 终止识别流并释放资源。
	Close() error
}

// ASRProvider 创建 ASR 流。
type ASRProvider interface {
	// StartStream 打开一个新的识别流。
	StartStream(ctx context.Context, cfg ASRConfig) (ASRStream, error)
}
