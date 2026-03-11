package pcm

import (
	"errors"
	"fmt"

	webrtcvad "github.com/maxhawkins/go-webrtcvad"
)

// VADMode 是 WebRTC VAD 的激进度级别。
// 值越高，越能过滤噪音，但也可能漏掉较轻的人声。
type VADMode int

const (
	// VADQuality 最低激进度，最大限度保留人声（可能误判噪音为人声）。
	VADQuality VADMode = 0
	// VADLowBitrate 低比特率模式。
	VADLowBitrate VADMode = 1
	// VADAggressive 激进模式，适合有一定噪音的环境。
	VADAggressive VADMode = 2
	// VADVeryAggressive 最高激进度，最适合嘈杂环境（如电话场景）。
	VADVeryAggressive VADMode = 3
)

// VAD 封装 WebRTC VAD，提供人声检测能力。
type VAD struct {
	vad        *webrtcvad.VAD
	sampleRate int
}

// NewVAD 创建新的 VAD 实例。
// sampleRate 支持 8000、16000、32000 Hz。
// mode 为激进度级别（0-3）。
func NewVAD(sampleRate int, mode VADMode) (*VAD, error) {
	vad, err := webrtcvad.New()
	if err != nil {
		return nil, fmt.Errorf("创建 WebRTC VAD: %w", err)
	}

	if err := vad.SetMode(int(mode)); err != nil {
		return nil, fmt.Errorf("设置 VAD mode %d: %w", mode, err)
	}

	return &VAD{
		vad:        vad,
		sampleRate: sampleRate,
	}, nil
}

// IsSpeech 检测音频帧是否包含人声。
// frame 必须是 16-bit PCM 单声道数据，长度对应 10/20/30ms。
// 8000 Hz 下：160 字节(10ms)、320 字节(20ms)、480 字节(30ms)。
func (v *VAD) IsSpeech(frame []byte) (bool, error) {
	if len(frame) == 0 {
		return false, errors.New("VAD process: 帧数据为空")
	}
	active, err := v.vad.Process(v.sampleRate, frame)
	if err != nil {
		return false, fmt.Errorf("VAD process: %w", err)
	}
	return active, nil
}

// ValidFrame 验证帧长度是否有效。
// WebRTC VAD 要求帧长度对应 10/20/30ms，参数为采样数（非字节数）。
func (v *VAD) ValidFrame(frame []byte) bool {
	sampleCount := len(frame) / 2 // 16-bit PCM，每采样 2 字节。
	return v.vad.ValidRateAndFrameLength(v.sampleRate, sampleCount)
}
