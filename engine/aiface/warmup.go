package aiface

import "context"

// Warmer 由支持连接预热的提供者实现。
// 调用 Warmup 可预建连接池、执行 WebSocket 握手或发送探测请求，
// 避免首次调用时的冷启动延迟（通常 100-300ms）。
//
// 实现示例：TTS 提供者在 Warmup 中建立 WebSocket 长连接；
// LLM 提供者在 Warmup 中发送空请求预热 HTTP/2 连接池。
//
// 非强制接口——不实现 Warmer 的提供者会被静默跳过。
type Warmer interface {
	Warmup(ctx context.Context) error
}
