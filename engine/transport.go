package engine

import "context"

// Transport 抽象音频的输入输出。
// 不同产品实现不同的传输层：
//   - 外呼产品：FreeSWITCH ESL（通过 uuid_broadcast 播放 WAV 文件）
//   - APP 产品：WebSocket（直接推送 PCM 帧）
//   - 音箱产品：蓝牙/WiFi 音频流
type Transport interface {
	// AudioIn 返回接收音频帧的通道。
	// 帧格式由产品决定（通常为 PCM16 LE）。
	// 通道关闭表示音频输入结束（对方挂断/断开）。
	AudioIn() <-chan []byte

	// PlayAudio 播放合成的音频数据。
	// 阻塞直到播放完成或 ctx 被取消（barge-in）。
	// audio 为 PCM16 LE 数据。
	PlayAudio(ctx context.Context, audio []byte) error

	// StopPlayback 立即停止当前播放（barge-in 场景）。
	StopPlayback(ctx context.Context) error

	// Close 关闭传输层，释放资源。
	Close() error
}
