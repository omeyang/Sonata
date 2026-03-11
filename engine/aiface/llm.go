package aiface

import "context"

// Message 表示发送给 LLM 的对话消息。
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// LLMConfig 持有 LLM 调用的配置。
type LLMConfig struct {
	Model       string
	MaxTokens   int
	Temperature float64
	TimeoutMs   int
}

// LLMProvider 生成流式文本响应。
type LLMProvider interface {
	// GenerateStream 返回响应 token 的通道。
	// 生成完成后通道关闭。
	GenerateStream(ctx context.Context, messages []Message, cfg LLMConfig) (<-chan string, error)

	// Generate 返回完整响应（非流式，用于后处理）。
	Generate(ctx context.Context, messages []Message, cfg LLMConfig) (string, error)
}
