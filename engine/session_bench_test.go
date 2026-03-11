package engine

import (
	"context"
	"encoding/binary"
	"runtime"
	"testing"
	"time"

	"github.com/omeyang/Sonata/engine/mediafsm"
)

// BenchmarkHandleAudioFrameWithResample 带重采样的完整帧处理（8→16kHz）。
func BenchmarkHandleAudioFrameWithResample(b *testing.B) {
	trans := newMockTransport()
	stream := newMockASRStream()
	s := New(Config{
		SessionID:       "bench",
		Transport:       trans,
		Transitions:     appTransitions(),
		SpeechDetector:  &mockSpeechDetector{isSpeech: false},
		InputSampleRate: 8000,
		ASRSampleRate:   16000,
	})
	s.startTime = time.Now()
	s.asrStream = stream
	_ = s.mfsm.Handle(mediafsm.EvAnswer)
	_ = s.mfsm.Handle(mediafsm.EvBotDone)

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	frame := silenceFrame160()

	b.SetBytes(int64(len(frame)))
	b.ResetTimer()

	for range b.N {
		s.handleAudioFrame(context.Background(), frame, timer)
	}
}

// BenchmarkHandleBargeInFrame 基准测试 barge-in 帧检测。
func BenchmarkHandleBargeInFrame(b *testing.B) {
	trans := newMockTransport()
	s := New(Config{
		SessionID:      "bench",
		Transport:      trans,
		Transitions:    appTransitions(),
		SpeechDetector: &mockSpeechDetector{isSpeech: false},
	})
	s.startTime = time.Now()
	_ = s.mfsm.Handle(mediafsm.EvAnswer)
	s.ttsPlaying.Store(true)

	frame := loudFrame(160, 10000)

	b.ResetTimer()
	for range b.N {
		s.handleBargeInFrame(context.Background(), frame)
	}
}

// BenchmarkSessionThroughput 模拟 1 秒通话（50 帧×20ms）的总吞吐量。
func BenchmarkSessionThroughput(b *testing.B) {
	frames := make([][]byte, 50)
	for i := range frames {
		frames[i] = makeFrame(160, int16(i*100))
	}

	trans := newMockTransport()
	stream := newMockASRStream()

	s := New(Config{
		SessionID:       "bench",
		Transport:       trans,
		Transitions:     appTransitions(),
		SpeechDetector:  &mockSpeechDetector{isSpeech: false},
		InputSampleRate: 8000,
		ASRSampleRate:   16000,
	})
	s.startTime = time.Now()
	s.asrStream = stream
	_ = s.mfsm.Handle(mediafsm.EvAnswer)
	_ = s.mfsm.Handle(mediafsm.EvBotDone)

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()

	b.ResetTimer()
	for range b.N {
		for _, f := range frames {
			s.handleAudioFrame(context.Background(), f, timer)
		}
	}
}

// BenchmarkConcurrentSessions 模拟并发 session 的帧处理。
func BenchmarkConcurrentSessions(b *testing.B) {
	frame := silenceFrame160()

	b.RunParallel(func(pb *testing.PB) {
		trans := newMockTransport()
		stream := newMockASRStream()

		s := New(Config{
			SessionID:       "bench",
			Transport:       trans,
			Transitions:     appTransitions(),
			SpeechDetector:  &mockSpeechDetector{isSpeech: false},
			InputSampleRate: 16000,
			ASRSampleRate:   16000,
		})
		s.startTime = time.Now()
		s.asrStream = stream
		_ = s.mfsm.Handle(mediafsm.EvAnswer)
		_ = s.mfsm.Handle(mediafsm.EvBotDone)

		timer := time.NewTimer(time.Hour)
		defer timer.Stop()

		for pb.Next() {
			s.handleAudioFrame(context.Background(), frame, timer)
		}
	})
}

// BenchmarkMemoryPressure 测量高频帧处理下的内存分配和 GC 影响。
func BenchmarkMemoryPressure(b *testing.B) {
	trans := newMockTransport()
	stream := newMockASRStream()

	s := New(Config{
		SessionID:       "bench",
		Transport:       trans,
		Transitions:     appTransitions(),
		SpeechDetector:  &mockSpeechDetector{isSpeech: false},
		InputSampleRate: 8000,
		ASRSampleRate:   16000,
	})
	s.startTime = time.Now()
	s.asrStream = stream
	_ = s.mfsm.Handle(mediafsm.EvAnswer)
	_ = s.mfsm.Handle(mediafsm.EvBotDone)

	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	frame := loudFrame(160, 5000)

	var mStart, mEnd runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&mStart)

	b.ResetTimer()
	for range b.N {
		s.handleAudioFrame(context.Background(), frame, timer)
	}
	b.StopTimer()

	runtime.ReadMemStats(&mEnd)
	b.ReportMetric(float64(mEnd.NumGC-mStart.NumGC), "gc_cycles")
	b.ReportMetric(float64(mEnd.PauseTotalNs-mStart.PauseTotalNs)/float64(b.N), "gc_pause_ns/op")
}

// makeFrame 生成测试用音频帧。
func makeFrame(samples int, amplitude int16) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(amplitude))
	}
	return buf
}
