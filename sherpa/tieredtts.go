package sherpa

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/omeyang/Sonata/engine/aiface"
)

// TieredTTSConfig 配置分层 TTS。
type TieredTTSConfig struct {
	// Local 本地 TTS（低延迟，短文本）。
	Local *LocalTTS

	// Cloud 云端 TTS（高质量，长文本）。
	Cloud aiface.TTSProvider

	// Threshold 文本字符数阈值。
	// 不超过此值使用本地 TTS，超过则使用云端 TTS。
	// 默认 10。
	Threshold int

	// Logger 日志记录器，为 nil 时使用 slog.Default()。
	Logger *slog.Logger
}

func (c *TieredTTSConfig) setDefaults() {
	if c.Threshold <= 0 {
		c.Threshold = 10
	}
}

// TieredTTS 根据文本长度自动选择本地或云端 TTS。
//
// 策略：
//   - 短文本（≤ Threshold 字符）→ 本地 TTS，消除网络延迟。
//   - 长文本（> Threshold 字符）→ 云端 TTS，获得更高语音质量。
//
// 典型场景：AI 回复的第一句话通常较短（如"好的"、"我理解了"），
// 使用本地 TTS 可以立即开始播放，显著降低首包延迟。
//
// 实现 aiface.TTSProvider 接口。
type TieredTTS struct {
	local     *LocalTTS
	cloud     aiface.TTSProvider
	threshold int
	logger    *slog.Logger
}

// NewTieredTTS 创建分层 TTS 实例。
func NewTieredTTS(cfg TieredTTSConfig) (*TieredTTS, error) {
	if cfg.Local == nil {
		return nil, errors.New("sherpa: TieredTTSConfig.Local 不能为 nil")
	}
	if cfg.Cloud == nil {
		return nil, errors.New("sherpa: TieredTTSConfig.Cloud 不能为 nil")
	}

	cfg.setDefaults()

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &TieredTTS{
		local:     cfg.Local,
		cloud:     cfg.Cloud,
		threshold: cfg.Threshold,
		logger:    logger,
	}, nil
}

// SynthesizeStream 根据每段文本长度选择本地或云端合成。
func (t *TieredTTS) SynthesizeStream(ctx context.Context, textCh <-chan string, cfg aiface.TTSConfig) (<-chan []byte, error) {
	audioCh := make(chan []byte, 8)

	go func() {
		defer close(audioCh)

		for {
			select {
			case <-ctx.Done():
				return
			case text, ok := <-textCh:
				if !ok {
					return
				}
				if text == "" {
					continue
				}

				pcm, err := t.synthesizeOne(ctx, text, cfg)
				if err != nil {
					t.logger.Warn("分层 TTS 合成失败",
						slog.String("error", err.Error()),
						slog.String("text", text),
					)
					return
				}
				if pcm == nil {
					continue
				}

				select {
				case audioCh <- pcm:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return audioCh, nil
}

// Synthesize 根据文本长度选择合成引擎。
func (t *TieredTTS) Synthesize(ctx context.Context, text string, cfg aiface.TTSConfig) ([]byte, error) {
	return t.synthesizeOne(ctx, text, cfg)
}

// Cancel 中止当前合成（仅转发给云端）。
func (t *TieredTTS) Cancel() error {
	if err := t.cloud.Cancel(); err != nil {
		return fmt.Errorf("sherpa: 云端 TTS 取消失败: %w", err)
	}
	return nil
}

// synthesizeOne 根据文本长度路由到本地或云端。
func (t *TieredTTS) synthesizeOne(ctx context.Context, text string, cfg aiface.TTSConfig) ([]byte, error) {
	runeCount := TextRuneCount(text)

	if runeCount <= t.threshold {
		t.logger.Debug("分层 TTS: 使用本地",
			slog.String("text", text),
			slog.Int("rune_count", runeCount),
		)
		return t.local.Synthesize(ctx, text, nil)
	}

	t.logger.Debug("分层 TTS: 使用云端",
		slog.String("text", text),
		slog.Int("rune_count", runeCount),
	)
	data, err := t.cloud.Synthesize(ctx, text, cfg)
	if err != nil {
		return nil, fmt.Errorf("sherpa: 云端 TTS 合成失败: %w", err)
	}
	return data, nil
}
