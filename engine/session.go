// Package engine 实现通用的实时语音对话会话运行时。
//
// Session 是 Sonata 的核心，驱动 ASR→LLM→TTS 流式管道，
// 管理听/说/打断的媒体节奏。它不关心音频从哪来（Transport 接口），
// 也不关心对话内容是什么（aiface.DialogEngine 接口）。
//
// 使用方式：
//
//	s := engine.New(cfg)
//	result, err := s.Run(ctx)
package engine

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/omeyang/Sonata/engine/aiface"
	"github.com/omeyang/Sonata/engine/mediafsm"
	"github.com/omeyang/Sonata/engine/pcm"
)

// SpeechDetector 检测音频帧是否包含人声。
// frame 为 PCM16 LE 单声道数据，长度应与采样率匹配（如 8kHz 20ms = 320 字节）。
type SpeechDetector interface {
	IsSpeech(frame []byte) (bool, error)
}

// Config 持有会话的所有依赖和配置。
type Config struct {
	// SessionID 是会话的唯一标识。
	SessionID string

	// Protection 保护参数。
	Protection ProtectionConfig

	// AI Provider。
	ASR    aiface.ASRProvider
	LLM    aiface.LLMProvider
	TTS    aiface.TTSProvider
	ASRCfg aiface.ASRConfig
	LLMCfg aiface.LLMConfig
	TTSCfg aiface.TTSConfig

	// Transport 音频传输层。
	Transport Transport

	// Dialogue 业务对话引擎。
	Dialogue aiface.DialogEngine

	// Transitions 媒体状态机转换规则。
	// 电话场景使用 mediafsm.PhoneTransitions()，APP 场景使用 mediafsm.AppTransitions()。
	Transitions []mediafsm.Transition

	// Logger 日志记录器。
	Logger *slog.Logger

	// Metrics 会话级度量收集（为 nil 时静默跳过）。
	Metrics Metrics

	// FillerAudios 填充词音频列表（预合成的 PCM16 数据）。
	// ASR 返回最终结果后，在 LLM 处理期间随机播放一个填充词，
	// 将感知延迟从 ~2s 降至 ~200ms。
	// 音频采样率应与 Transport 期望的输出格式一致。
	// 为空时不播放填充词。
	FillerAudios [][]byte

	// InputSampleRate 输入音频的采样率（Hz）。
	// 用于 VAD 初始化和能量检测。默认为 8000。
	InputSampleRate int

	// ASRSampleRate ASR 需要的采样率（Hz）。
	// 如果与 InputSampleRate 不同，会自动重采样。默认为 16000。
	ASRSampleRate int

	// SpeechDetector 人声检测器（可选）。
	// 为 nil 时退回到基于 EnergyThresholdDBFS 的纯能量检测。
	// 典型实现：pcm.VAD（WebRTC）、sherpa.SileroVAD 等。
	SpeechDetector SpeechDetector

	// EnergyThresholdDBFS SpeechDetector 不可用时的能量阈值退回方案。
	// 默认为 -35.0 dBFS。
	EnergyThresholdDBFS float64
}

// ProtectionConfig 通话/会话保护参数。
type ProtectionConfig struct {
	MaxDurationSec         int // 最大会话时长（秒）。
	MaxSilenceSec          int // 最大静默时间（秒）。
	FirstSilenceTimeoutSec int // 首次静默超时（秒）。
}

// Result 是已完成会话的结果。
type Result struct {
	SessionID string          `json:"session_id"`
	Duration  int             `json:"duration_sec"`
	Events    []RecordedEvent `json:"events"`
}

// Session 编排单个实时语音对话。
// 管理媒体 FSM、对话引擎和流式 ASR→LLM→TTS 管线。
type Session struct {
	cfg    Config
	mfsm   *mediafsm.FSM
	logger *slog.Logger
	ctx    context.Context //nolint:containedctx // Run() 传入，供异步方法使用

	mu        sync.Mutex
	events    []RecordedEvent
	startTime time.Time

	silenceCount int

	// 流式 ASR 会话。
	asrStream  aiface.ASRStream
	asrResults chan aiface.ASREvent

	// TTS 播放完成信号。
	botDoneCh chan struct{}

	// barge-in 连续活跃帧计数，防止噪音误触发。
	bargeInFrames int
	// 取消当前 TTS 合成/播放。
	ttsCancel context.CancelFunc
	// ttsPlaying 标记 TTS 是否正在播放。
	ttsPlaying atomic.Bool

	// speechDetector 检测人声，由 Config.SpeechDetector 注入。
	speechDetector SpeechDetector

	// resampleBuf 重采样输出缓冲区复用（单 goroutine 内安全复用，避免每帧分配）。
	resampleBuf []byte
}

// New 创建新的语音对话会话。
func New(cfg Config) *Session {
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	logger = logger.With(slog.String("session_id", cfg.SessionID))

	if cfg.InputSampleRate <= 0 {
		cfg.InputSampleRate = 8000
	}
	if cfg.ASRSampleRate <= 0 {
		cfg.ASRSampleRate = 16000
	}
	if cfg.EnergyThresholdDBFS == 0 {
		cfg.EnergyThresholdDBFS = -35.0
	}
	if cfg.Protection.MaxDurationSec <= 0 {
		cfg.Protection.MaxDurationSec = 300
	}
	if cfg.Protection.MaxSilenceSec <= 0 {
		cfg.Protection.MaxSilenceSec = 15
	}
	if cfg.Protection.FirstSilenceTimeoutSec <= 0 {
		cfg.Protection.FirstSilenceTimeoutSec = 6
	}

	s := &Session{
		cfg:            cfg,
		mfsm:           mediafsm.NewFSM(mediafsm.Idle, cfg.Transitions, mediafsm.Unsynced()),
		logger:         logger,
		asrResults:     make(chan aiface.ASREvent, 16),
		botDoneCh:      make(chan struct{}, 1),
		speechDetector: cfg.SpeechDetector,
	}

	return s
}

// Run 执行完整的对话生命周期。阻塞直到会话完成。
// 调用方负责在 Run 之前完成会话建立（如电话拨号、WebSocket 握手），
// 并在建立成功后调用 Run。
func (s *Session) Run(ctx context.Context) (*Result, error) {
	s.startTime = time.Now()

	ctx, cancel := context.WithTimeout(ctx, time.Duration(s.cfg.Protection.MaxDurationSec)*time.Second)
	defer cancel()

	s.ctx = ctx

	// 预热提供者连接，避免首次调用冷启动延迟。
	s.warmupProviders(ctx)

	s.mfsm.OnTransition(func(from, to mediafsm.State, event mediafsm.Event) {
		s.logger.Info("media transition",
			slog.String("from", from.String()),
			slog.String("to", to.String()),
			slog.String("event", event.String()),
		)
	})

	// 开始对话：播放开场白。
	s.startDialogue()

	// 主事件循环。
	err := s.eventLoop(ctx)

	return s.buildResult(), err
}

// warmupProviders 预热实现了 Warmer 接口的提供者。
// 预建连接池或执行握手，减少首次调用的冷启动延迟。
func (s *Session) warmupProviders(ctx context.Context) {
	providers := []any{s.cfg.ASR, s.cfg.LLM, s.cfg.TTS}
	for _, p := range providers {
		if w, ok := p.(aiface.Warmer); ok {
			if err := w.Warmup(ctx); err != nil {
				s.logger.Warn("provider warmup 失败", slog.String("error", err.Error()))
			}
		}
	}
}

// FSM 返回媒体状态机，供产品层在 Run 之前驱动前置状态（如拨号/振铃/AMD）。
func (s *Session) FSM() *mediafsm.FSM {
	return s.mfsm
}

// RecordEvent 将事件追加到会话事件日志（供产品层记录自定义事件）。
func (s *Session) RecordEvent(eventType EventType, metadata map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.events = append(s.events, RecordedEvent{
		EventType:   eventType,
		TimestampMs: time.Since(s.startTime).Milliseconds(),
		Metadata:    metadata,
	})
}

// eventLoop 处理音频和事件直到会话结束。
func (s *Session) eventLoop(ctx context.Context) error {
	silenceTimer := time.NewTimer(time.Duration(s.cfg.Protection.FirstSilenceTimeoutSec) * time.Second)
	defer silenceTimer.Stop()

	audioIn := s.cfg.Transport.AudioIn()

	for {
		select {
		case <-ctx.Done():
			s.logger.Info("session context done", slog.String("reason", ctx.Err().Error()))
			s.handleHangup("max_duration")
			return fmt.Errorf("session context done: %w", ctx.Err())

		case frame, ok := <-audioIn:
			if !ok {
				s.handleHangup("audio_closed")
				return nil
			}
			s.handleAudioFrame(ctx, frame, silenceTimer)

		case <-silenceTimer.C:
			s.handleSilenceTimeout(silenceTimer)
			if s.mfsm.IsTerminal() {
				return nil
			}

		case asrEvt := <-s.asrResults:
			s.handleStreamingASR(ctx, asrEvt, silenceTimer)
			if s.mfsm.IsTerminal() {
				return nil
			}

		case <-s.botDoneCh:
			s.tryHandleFSMEvent(mediafsm.EvBotDone, "handle bot done")
			s.RecordEvent(EventBotSpeakEnd, nil)
			silenceTimer.Reset(time.Duration(s.cfg.Protection.MaxSilenceSec) * time.Second)
		}
	}
}

// handleAudioFrame 根据当前媒体状态分发音频帧。
func (s *Session) handleAudioFrame(ctx context.Context, frame []byte, silenceTimer *time.Timer) {
	state := s.mfsm.State()

	switch state {
	case mediafsm.WaitingUser:
		s.feedASR(ctx, frame)
		s.handleWaitingUserFrame(frame, silenceTimer)
	case mediafsm.UserSpeaking:
		s.feedASR(ctx, frame)
	case mediafsm.BotSpeaking:
		s.handleBargeInFrame(ctx, frame)
	default:
		// 其他状态下不处理音频。
	}
}

// handleWaitingUserFrame 在等待用户说话时检测语音起始。
func (s *Session) handleWaitingUserFrame(frame []byte, silenceTimer *time.Timer) {
	if !s.isSpeechFrame(frame) {
		return
	}

	s.handleFSMEvent(mediafsm.EvSpeechStart, "handle speech start")
	s.RecordEvent(EventUserSpeechStart, nil)
	silenceTimer.Reset(time.Duration(s.cfg.Protection.MaxSilenceSec) * time.Second)
}

// bargeInThreshold 是触发 barge-in 需要的连续活跃帧数（约 200ms）。
const bargeInThreshold = 10

// handleBargeInFrame 在 AI 说话时检测用户打断。
func (s *Session) handleBargeInFrame(ctx context.Context, frame []byte) {
	if !s.ttsPlaying.Load() {
		return
	}

	if !s.isSpeechFrame(frame) {
		s.bargeInFrames = 0
		return
	}

	s.bargeInFrames++
	if s.bargeInFrames < bargeInThreshold {
		return
	}
	s.bargeInFrames = 0

	s.logger.Info("barge-in 触发（VAD 检测到人声）")
	s.ttsPlaying.Store(false)
	s.handleFSMEvent(mediafsm.EvBargeIn, "handle barge-in")
	s.RecordEvent(EventBargeIn, nil)

	// 取消进行中的 TTS 合成。
	if s.ttsCancel != nil {
		s.ttsCancel()
		s.ttsCancel = nil
	}

	// 停止传输层播放。
	if err := s.cfg.Transport.StopPlayback(ctx); err != nil {
		s.logger.Warn("stop playback", slog.String("error", err.Error()))
	}

	s.handleFSMEvent(mediafsm.EvBargeInDone, "handle barge-in done")
	s.RecordEvent(EventUserSpeechStart, nil)
}

// isSpeechFrame 判断音频帧是否包含人声。
// 优先使用注入的 SpeechDetector，不可用时退回能量阈值检测。
func (s *Session) isSpeechFrame(frame []byte) bool {
	if s.speechDetector != nil {
		speech, err := s.speechDetector.IsSpeech(frame)
		if err == nil {
			return speech
		}
		s.logger.Debug("SpeechDetector 处理失败，退回能量检测", slog.String("error", err.Error()))
	}
	return pcm.EnergyDBFS(frame) > s.cfg.EnergyThresholdDBFS
}

// handleFSMEvent 向媒体 FSM 发送事件。
func (s *Session) handleFSMEvent(event mediafsm.Event, msg string) {
	if err := s.mfsm.Handle(event); err != nil {
		s.logger.Warn(msg, slog.String("error", err.Error()))
	}
}

// tryHandleFSMEvent 仅在事件可处理时发送。
func (s *Session) tryHandleFSMEvent(event mediafsm.Event, msg string) {
	if !s.mfsm.CanHandle(event) {
		return
	}
	if err := s.mfsm.Handle(event); err != nil {
		s.logger.Warn(msg, slog.String("error", err.Error()))
	}
}

// startDialogue 播放开场白并启动 ASR。
func (s *Session) startDialogue() {
	if s.cfg.Dialogue == nil {
		return
	}

	opening := s.cfg.Dialogue.Opening()
	s.logger.Info("对话开始", slog.String("text", opening))
	s.RecordEvent(EventBotSpeakStart, map[string]string{"text": opening})

	// 启动 ASR 流。
	s.startASRStream()

	// 有 TTS 时异步合成并播放。
	if s.cfg.TTS != nil && s.ctx != nil {
		s.synthesizeAndPlayAsync(opening)
		return
	}

	// 无 TTS 时直接完成（测试兼容）。
	s.RecordEvent(EventBotSpeakEnd, nil)
	s.tryHandleFSMEvent(mediafsm.EvBotDone, "handle bot done (no TTS)")
}

// handleStreamingASR 处理流式 ASR 事件。
func (s *Session) handleStreamingASR(ctx context.Context, evt aiface.ASREvent, silenceTimer *time.Timer) {
	if !evt.IsFinal || evt.Text == "" {
		return
	}

	state := s.mfsm.State()
	if state != mediafsm.WaitingUser && state != mediafsm.UserSpeaking {
		s.logger.Debug("ASR 结果丢弃（非用户状态）",
			slog.String("text", evt.Text),
			slog.String("state", state.String()),
		)
		return
	}

	s.logger.Info("ASR 最终结果", slog.String("text", evt.Text), slog.Float64("confidence", evt.Confidence))

	s.RecordEvent(EventUserSpeechEnd, map[string]string{"text": evt.Text})
	s.tryHandleFSMEvent(mediafsm.EvSpeechEnd, "handle speech end")

	if s.cfg.Dialogue == nil {
		s.tryHandleFSMEvent(mediafsm.EvProcessingDone, "handle processing done (no engine)")
		return
	}

	// 流式管道：LLM 流式生成 → 按句拆分 → 逐句 TTS → 逐句播放。
	sentenceCh, err := s.cfg.Dialogue.ProcessStream(ctx, evt.Text)
	if err != nil {
		s.logger.Error("对话流式处理失败", slog.String("error", err.Error()))
		s.tryHandleFSMEvent(mediafsm.EvProcessingDone, "handle processing done (error)")
		s.tryHandleFSMEvent(mediafsm.EvBotDone, "handle bot done (error)")
		return
	}

	s.RecordEvent(EventBotSpeakStart, map[string]string{"text": "(streaming)"})
	s.tryHandleFSMEvent(mediafsm.EvProcessingDone, "handle processing done")

	if s.cfg.TTS != nil {
		s.synthesizeAndPlayStreamAsync(sentenceCh, nil)
	} else {
		go func() {
			for range sentenceCh {
			}
		}()
		s.tryHandleFSMEvent(mediafsm.EvBotDone, "handle bot done (no TTS)")
		s.RecordEvent(EventBotSpeakEnd, nil)
		silenceTimer.Reset(time.Duration(s.cfg.Protection.MaxSilenceSec) * time.Second)
	}
}

// handleSilenceTimeout 处理静默超时。
func (s *Session) handleSilenceTimeout(silenceTimer *time.Timer) {
	s.silenceCount++
	s.RecordEvent(EventSilenceTimeout, map[string]string{
		"count": strconv.Itoa(s.silenceCount),
	})

	s.tryHandleFSMEvent(mediafsm.EvSilenceTimeout, "handle silence timeout")

	if s.silenceCount >= 2 {
		s.tryHandleFSMEvent(mediafsm.EvSecondSilence, "handle second silence")
		return
	}

	// 首次静默超时：播放提示语后回到等待用户。
	// FSM: SilenceTimeout → (EvSilencePromptDone) → BotSpeaking → (EvBotDone) → WaitingUser。
	s.tryHandleFSMEvent(mediafsm.EvSilencePromptDone, "handle silence prompt done")

	// 无 TTS 时直接完成播放。有 TTS 时由 synthesizeAndPlayAsync 触发 botDoneCh。
	if s.cfg.TTS == nil {
		s.tryHandleFSMEvent(mediafsm.EvBotDone, "handle bot done (silence prompt, no TTS)")
	}

	silenceTimer.Reset(time.Duration(s.cfg.Protection.MaxSilenceSec) * time.Second)
}

// handleHangup 转换到挂断状态。
func (s *Session) handleHangup(cause string) {
	s.logger.Info("session hangup", slog.String("cause", cause))

	s.tryHandleFSMEvent(mediafsm.EvHangup, "handle hangup")
	s.RecordEvent(EventHangup, map[string]string{"cause": cause})
}

// feedASR 将音频帧送入流式 ASR（必要时重采样）。
func (s *Session) feedASR(ctx context.Context, frame []byte) {
	if s.asrStream == nil {
		return
	}

	var data []byte
	if s.cfg.InputSampleRate == 8000 && s.cfg.ASRSampleRate == 16000 {
		s.resampleBuf = pcm.Resample8to16Into(s.resampleBuf, frame)
		data = s.resampleBuf
	} else {
		data = frame
	}

	if data == nil {
		return
	}
	if err := s.asrStream.FeedAudio(ctx, data); err != nil {
		s.logger.Warn("ASR feed 失败", slog.String("error", err.Error()))
	}
}

// startASRStream 初始化流式 ASR 并转发事件。
func (s *Session) startASRStream() {
	if s.cfg.ASR == nil || s.ctx == nil {
		return
	}

	stream, err := s.cfg.ASR.StartStream(s.ctx, s.cfg.ASRCfg)
	if err != nil {
		s.logger.Error("ASR 流启动失败", slog.String("error", err.Error()))
		return
	}
	s.asrStream = stream

	go func() {
		for evt := range stream.Events() {
			select {
			case s.asrResults <- evt:
			default:
				s.logger.Warn("ASR 事件通道已满，丢弃事件")
			}
		}
	}()
}

// synthesizeAndPlayAsync 异步合成 TTS 并通过 Transport 播放。
func (s *Session) synthesizeAndPlayAsync(text string) {
	if s.ttsCancel != nil {
		s.ttsCancel()
	}
	ttsCtx, cancel := context.WithCancel(s.ctx)
	s.ttsCancel = cancel

	go func() {
		defer func() {
			select {
			case s.botDoneCh <- struct{}{}:
			default:
			}
		}()

		s.logger.Info("TTS 合成开始", slog.String("text", text))
		audioData, err := s.cfg.TTS.Synthesize(ttsCtx, text, s.cfg.TTSCfg)
		if err != nil {
			if ttsCtx.Err() != nil {
				s.logger.Info("TTS 合成被取消")
				return
			}
			s.logger.Error("TTS 合成失败", slog.String("error", err.Error()))
			return
		}

		if len(audioData) == 0 {
			return
		}

		s.ttsPlaying.Store(true)
		if playErr := s.cfg.Transport.PlayAudio(ttsCtx, audioData); playErr != nil {
			if ttsCtx.Err() == nil {
				s.logger.Error("播放失败", slog.String("error", playErr.Error()))
			}
		}
		s.ttsPlaying.Store(false)
	}()
}

// synthesizeAndPlayStreamAsync 流式合成 TTS 并逐句播放。
// onComplete 在所有句段播放完成后调用（可为 nil），用于预推理轮次确认。
func (s *Session) synthesizeAndPlayStreamAsync(sentenceCh <-chan string, onComplete func()) {
	if s.ttsCancel != nil {
		s.ttsCancel()
	}
	ttsCtx, cancel := context.WithCancel(s.ctx)
	s.ttsCancel = cancel

	go func() {
		defer func() {
			s.ttsPlaying.Store(false)
			if onComplete != nil {
				onComplete()
			}
			select {
			case s.botDoneCh <- struct{}{}:
			default:
			}
		}()

		// 播放填充词，掩盖 LLM 处理延迟。
		s.playFiller(ttsCtx)

		for sentence := range sentenceCh {
			if ttsCtx.Err() != nil {
				return
			}

			s.logger.Info("TTS 句段合成开始", slog.String("text", sentence))
			audioData, err := s.cfg.TTS.Synthesize(ttsCtx, sentence, s.cfg.TTSCfg)
			if err != nil {
				if ttsCtx.Err() != nil {
					return
				}
				s.logger.Error("TTS 句段合成失败", slog.String("error", err.Error()))
				continue
			}

			if len(audioData) == 0 {
				continue
			}

			s.ttsPlaying.Store(true)
			if playErr := s.cfg.Transport.PlayAudio(ttsCtx, audioData); playErr != nil {
				if ttsCtx.Err() != nil {
					return
				}
				s.logger.Error("句段播放失败", slog.String("error", playErr.Error()))
			}
		}
	}()
}

// fillerIndex 用于轮询选择填充词，避免使用随机数。
var fillerIndex atomic.Int64

// playFiller 播放填充词音频，掩盖 LLM 处理延迟。
// 填充词在 ASR 返回最终结果后立即播放，用户感知延迟从 ~2s 降至 ~200ms。
func (s *Session) playFiller(ctx context.Context) {
	if len(s.cfg.FillerAudios) == 0 {
		return
	}

	idx := int(fillerIndex.Add(1)-1) % len(s.cfg.FillerAudios)
	filler := s.cfg.FillerAudios[idx]
	if len(filler) == 0 {
		return
	}

	s.ttsPlaying.Store(true)
	if err := s.cfg.Transport.PlayAudio(ctx, filler); err != nil {
		if ctx.Err() == nil {
			s.logger.Warn("填充词播放失败", slog.String("error", err.Error()))
		}
	}

	if s.cfg.Metrics != nil {
		s.cfg.Metrics.IncFillerPlayed()
	}
}

// buildResult 构造最终会话结果。
func (s *Session) buildResult() *Result {
	s.mu.Lock()
	events := make([]RecordedEvent, len(s.events))
	copy(events, s.events)
	s.mu.Unlock()

	return &Result{
		SessionID: s.cfg.SessionID,
		Duration:  int(time.Since(s.startTime).Seconds()),
		Events:    events,
	}
}
