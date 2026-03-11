package sherpa

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	"github.com/omeyang/Sonata/engine/aiface"
)

// RacingASRConfig 配置竞速 ASR。
type RacingASRConfig struct {
	// Local 本地 ASR（低延迟端点检测）。
	Local aiface.ASRProvider

	// Cloud 云端 ASR（高精度识别）。
	Cloud aiface.ASRProvider

	// Logger 日志记录器，为 nil 时使用 slog.Default()。
	Logger *slog.Logger
}

// RacingASR 同时启动本地和云端 ASR 流，取先到的 final 结果。
//
// 策略：
//   - 音频同时喂入本地和云端两个流。
//   - Partial 事件优先转发本地（低延迟），如果本地无 partial 则转发云端。
//   - Final 事件取先到者，另一个流的 final 丢弃。
//   - 任一流出错不影响另一个流的正常工作。
//
// 实现 aiface.ASRProvider 接口。
type RacingASR struct {
	local  aiface.ASRProvider
	cloud  aiface.ASRProvider
	logger *slog.Logger
}

// NewRacingASR 创建竞速 ASR 实例。
func NewRacingASR(cfg RacingASRConfig) (*RacingASR, error) {
	if cfg.Local == nil {
		return nil, errors.New("sherpa: RacingASRConfig.Local 不能为 nil")
	}
	if cfg.Cloud == nil {
		return nil, errors.New("sherpa: RacingASRConfig.Cloud 不能为 nil")
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &RacingASR{
		local:  cfg.Local,
		cloud:  cfg.Cloud,
		logger: logger,
	}, nil
}

// StartStream 同时打开本地和云端识别流，返回竞速合并流。
func (r *RacingASR) StartStream(ctx context.Context, cfg aiface.ASRConfig) (aiface.ASRStream, error) {
	localStream, err := r.local.StartStream(ctx, cfg)
	if err != nil {
		return nil, errors.Join(errors.New("sherpa: 本地 ASR 流创建失败"), err)
	}

	cloudStream, err := r.cloud.StartStream(ctx, cfg)
	if err != nil {
		_ = localStream.Close()
		return nil, errors.Join(errors.New("sherpa: 云端 ASR 流创建失败"), err)
	}

	ctx, cancel := context.WithCancel(ctx)
	s := &racingASRStream{
		local:  localStream,
		cloud:  cloudStream,
		events: make(chan aiface.ASREvent, 32),
		logger: r.logger,
		cancel: cancel,
	}

	go s.mergeLoop(ctx)

	return s, nil
}

// racingASRStream 合并本地和云端两个 ASR 流的事件。
type racingASRStream struct {
	local  aiface.ASRStream
	cloud  aiface.ASRStream
	events chan aiface.ASREvent
	logger *slog.Logger
	cancel context.CancelFunc

	mu     sync.Mutex
	closed bool
}

// FeedAudio 同时向两个流发送音频数据。
// 任一流出错时记录日志但不中断另一个流。
func (s *racingASRStream) FeedAudio(ctx context.Context, chunk []byte) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return errors.New("sherpa: Racing ASR 流已关闭")
	}
	s.mu.Unlock()

	var localErr, cloudErr error

	// 并行喂入音频。
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		localErr = s.local.FeedAudio(ctx, chunk)
	}()
	go func() {
		defer wg.Done()
		cloudErr = s.cloud.FeedAudio(ctx, chunk)
	}()

	wg.Wait()

	if localErr != nil && cloudErr != nil {
		return errors.Join(
			errors.New("sherpa: 两个 ASR 流均失败"),
			localErr, cloudErr,
		)
	}
	if localErr != nil {
		s.logger.Warn("本地 ASR 流喂入失败", slog.String("error", localErr.Error()))
	}
	if cloudErr != nil {
		s.logger.Warn("云端 ASR 流喂入失败", slog.String("error", cloudErr.Error()))
	}

	return nil
}

// Events 返回合并后的事件通道。
func (s *racingASRStream) Events() <-chan aiface.ASREvent {
	return s.events
}

// Close 关闭两个底层流并停止合并循环。
func (s *racingASRStream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()

	s.cancel()

	localErr := s.local.Close()
	cloudErr := s.cloud.Close()

	return errors.Join(localErr, cloudErr)
}

// mergeLoop 从两个流读取事件并合并输出。
// 策略：
//   - partial 优先转发本地（延迟低），如果仅云端有 partial 也转发。
//   - final 取先到者，后到者丢弃。
func (s *racingASRStream) mergeLoop(ctx context.Context) {
	defer close(s.events)

	localCh := s.local.Events()
	cloudCh := s.cloud.Events()

	var finalSent bool

	for localCh != nil || cloudCh != nil {
		select {
		case <-ctx.Done():
			return

		case evt, ok := <-localCh:
			if !ok {
				localCh = nil
				continue
			}
			if evt.IsFinal {
				if finalSent {
					continue
				}
				finalSent = true
				s.logger.Info("竞速 ASR: 本地 final 胜出",
					slog.String("text", evt.Text),
					slog.Int("latency_ms", evt.LatencyMs),
				)
			}
			s.send(ctx, evt)

		case evt, ok := <-cloudCh:
			if !ok {
				cloudCh = nil
				continue
			}
			if evt.IsFinal {
				if finalSent {
					continue
				}
				finalSent = true
				s.logger.Info("竞速 ASR: 云端 final 胜出",
					slog.String("text", evt.Text),
					slog.Int("latency_ms", evt.LatencyMs),
				)
			}
			s.send(ctx, evt)
		}
	}
}

// send 发送事件到合并通道，partial 事件允许丢弃。
func (s *racingASRStream) send(ctx context.Context, evt aiface.ASREvent) {
	if evt.IsFinal {
		select {
		case s.events <- evt:
		case <-ctx.Done():
		}
	} else {
		select {
		case s.events <- evt:
		default:
			// partial 通道满时丢弃（不阻塞合并循环）。
		}
	}
}
