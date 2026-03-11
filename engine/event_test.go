package engine

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEventTypeConstants 验证所有事件类型常量值。
func TestEventTypeConstants(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		event EventType
		want  string
	}{
		{"用户语音开始", EventUserSpeechStart, "user_speech_start"},
		{"用户语音结束", EventUserSpeechEnd, "user_speech_end"},
		{"AI说话开始", EventBotSpeakStart, "bot_speak_start"},
		{"AI说话结束", EventBotSpeakEnd, "bot_speak_end"},
		{"打断", EventBargeIn, "barge_in"},
		{"静默超时", EventSilenceTimeout, "silence_timeout"},
		{"ASR错误", EventASRError, "asr_error"},
		{"LLM超时", EventLLMTimeout, "llm_timeout"},
		{"TTS错误", EventTTSError, "tts_error"},
		{"挂断", EventHangup, "hangup"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, EventType(tt.want), tt.event)
			assert.Equal(t, tt.want, string(tt.event))
		})
	}
}

// TestEventTypeUniqueness 验证所有事件类型互不相同。
func TestEventTypeUniqueness(t *testing.T) {
	t.Parallel()

	all := []EventType{
		EventUserSpeechStart, EventUserSpeechEnd,
		EventBotSpeakStart, EventBotSpeakEnd,
		EventBargeIn, EventSilenceTimeout,
		EventASRError, EventLLMTimeout, EventTTSError,
		EventHangup,
	}

	seen := make(map[EventType]bool)
	for _, e := range all {
		if seen[e] {
			t.Errorf("重复的事件类型: %s", e)
		}
		seen[e] = true
	}
	assert.Len(t, seen, 10, "应有 10 个不同的事件类型")
}

// TestRecordedEventJSON 验证 RecordedEvent 的 JSON 序列化和反序列化。
func TestRecordedEventJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		event RecordedEvent
	}{
		{
			name: "带元数据",
			event: RecordedEvent{
				EventType:   EventUserSpeechEnd,
				TimestampMs: 1500,
				Metadata:    map[string]string{"text": "你好"},
			},
		},
		{
			name: "无元数据",
			event: RecordedEvent{
				EventType:   EventHangup,
				TimestampMs: 30000,
				Metadata:    nil,
			},
		},
		{
			name: "零时间戳",
			event: RecordedEvent{
				EventType:   EventBargeIn,
				TimestampMs: 0,
			},
		},
		{
			name: "空元数据",
			event: RecordedEvent{
				EventType:   EventBotSpeakStart,
				TimestampMs: 500,
				Metadata:    map[string]string{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			data, err := json.Marshal(tt.event)
			require.NoError(t, err, "JSON 序列化不应失败")

			var got RecordedEvent
			err = json.Unmarshal(data, &got)
			require.NoError(t, err, "JSON 反序列化不应失败")

			assert.Equal(t, tt.event.EventType, got.EventType)
			assert.Equal(t, tt.event.TimestampMs, got.TimestampMs)
		})
	}
}

// TestRecordedEventJSONOmitEmpty 验证 nil Metadata 在 JSON 中被省略。
func TestRecordedEventJSONOmitEmpty(t *testing.T) {
	t.Parallel()

	event := RecordedEvent{
		EventType:   EventHangup,
		TimestampMs: 1000,
		Metadata:    nil,
	}

	data, err := json.Marshal(event)
	require.NoError(t, err)
	assert.NotContains(t, string(data), "metadata")
}

// TestRecordedEventJSONFields 验证 JSON 字段名映射正确。
func TestRecordedEventJSONFields(t *testing.T) {
	t.Parallel()

	event := RecordedEvent{
		EventType:   EventUserSpeechStart,
		TimestampMs: 42,
		Metadata:    map[string]string{"key": "val"},
	}

	data, err := json.Marshal(event)
	require.NoError(t, err)

	var m map[string]any
	err = json.Unmarshal(data, &m)
	require.NoError(t, err)

	assert.Contains(t, m, "event_type")
	assert.Contains(t, m, "timestamp_ms")
	assert.Contains(t, m, "metadata")
	assert.Equal(t, "user_speech_start", m["event_type"])
	assert.Equal(t, float64(42), m["timestamp_ms"])
}

// FuzzEventTypeString 模糊测试 EventType 到字符串的转换不会 panic。
func FuzzEventTypeString(f *testing.F) {
	f.Add("user_speech_start")
	f.Add("hangup")
	f.Add("")
	f.Add("unknown_event")
	f.Add("中文事件")

	f.Fuzz(func(t *testing.T, s string) {
		e := EventType(s)
		got := string(e)
		if got != s {
			t.Errorf("EventType(%q) 转字符串后不一致: %q", s, got)
		}
	})
}
