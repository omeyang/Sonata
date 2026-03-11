package engine

// EventType 表示会话中记录的事件类型。
type EventType string

// 会话事件类型。
const (
	EventUserSpeechStart EventType = "user_speech_start"
	EventUserSpeechEnd   EventType = "user_speech_end"
	EventBotSpeakStart   EventType = "bot_speak_start"
	EventBotSpeakEnd     EventType = "bot_speak_end"
	EventBargeIn         EventType = "barge_in"
	EventSilenceTimeout  EventType = "silence_timeout"
	EventASRError        EventType = "asr_error"
	EventLLMTimeout      EventType = "llm_timeout"
	EventTTSError        EventType = "tts_error"
	EventHangup          EventType = "hangup"
)

// RecordedEvent 记录带时间戳的事件，用于会话过程跟踪。
type RecordedEvent struct {
	EventType   EventType         `json:"event_type"`
	TimestampMs int64             `json:"timestamp_ms"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}
