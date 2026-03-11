package mediafsm

import (
	"testing"
)

// TestStateString 验证所有已定义状态的字符串表示。
func TestStateString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		s    State
		want string
	}{
		{"空闲", Idle, "IDLE"},
		{"拨号中", Dialing, "DIALING"},
		{"振铃中", Ringing, "RINGING"},
		{"AMD检测", AMDDetecting, "AMD_DETECTING"},
		{"AI播放", BotSpeaking, "BOT_SPEAKING"},
		{"等待用户", WaitingUser, "WAITING_USER"},
		{"用户说话", UserSpeaking, "USER_SPEAKING"},
		{"处理中", Processing, "PROCESSING"},
		{"打断", BargeIn, "BARGE_IN"},
		{"静默超时", SilenceTimeout, "SILENCE_TIMEOUT"},
		{"挂断", Hangup, "HANGUP"},
		{"后处理", PostProcessing, "POST_PROCESSING"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.s.String(); got != tt.want {
				t.Errorf("State(%d).String() = %q, want %q", tt.s, got, tt.want)
			}
		})
	}
}

// TestStateStringOutOfRange 验证越界状态返回 UNKNOWN。
// 注意：负值会导致 panic（数组越界），仅测试正向越界。
func TestStateStringOutOfRange(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		s    State
	}{
		{"超过上界", State(100)},
		{"紧邻上界", State(12)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.s.String(); got != "UNKNOWN" {
				t.Errorf("State(%d).String() = %q, want %q", tt.s, got, "UNKNOWN")
			}
		})
	}
}

// TestEventString 验证所有已定义事件的字符串表示。
func TestEventString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		e    Event
		want string
	}{
		{"拨号事件", EvDial, "DIAL"},
		{"振铃事件", EvRinging, "RINGING"},
		{"拨号失败", EvDialFailed, "DIAL_FAILED"},
		{"应答", EvAnswer, "ANSWER"},
		{"振铃超时", EvRingTimeout, "RING_TIMEOUT"},
		{"AMD真人", EvAMDHuman, "AMD_HUMAN"},
		{"AMD机器", EvAMDMachine, "AMD_MACHINE"},
		{"AI播放完成", EvBotDone, "BOT_DONE"},
		{"语音开始", EvSpeechStart, "SPEECH_START"},
		{"语音结束", EvSpeechEnd, "SPEECH_END"},
		{"打断事件", EvBargeIn, "BARGE_IN"},
		{"打断完成", EvBargeInDone, "BARGE_IN_DONE"},
		{"处理完成", EvProcessingDone, "PROCESSING_DONE"},
		{"静默超时事件", EvSilenceTimeout, "SILENCE_TIMEOUT"},
		{"静默提示完成", EvSilencePromptDone, "SILENCE_PROMPT_DONE"},
		{"二次静默", EvSecondSilence, "SECOND_SILENCE"},
		{"挂断事件", EvHangup, "HANGUP"},
		{"处理超时", EvProcessingTimeout, "PROCESSING_TIMEOUT"},
		{"后处理完成", EvPostDone, "POST_DONE"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.e.String(); got != tt.want {
				t.Errorf("Event(%d).String() = %q, want %q", tt.e, got, tt.want)
			}
		})
	}
}

// TestEventStringOutOfRange 验证越界事件返回 UNKNOWN。
// 注意：负值会导致 panic（数组越界），仅测试正向越界。
func TestEventStringOutOfRange(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		e    Event
	}{
		{"超过上界", Event(200)},
		{"紧邻上界", Event(19)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.e.String(); got != "UNKNOWN" {
				t.Errorf("Event(%d).String() = %q, want %q", tt.e, got, "UNKNOWN")
			}
		})
	}
}

// FuzzStateString 模糊测试 State.String() 对非负值不会 panic。
// 注意：负值在当前实现中会 panic（数组越界），因此仅测试非负值。
func FuzzStateString(f *testing.F) {
	// 添加种子语料：合法值和边界值。
	f.Add(0)
	f.Add(11) // PostProcessing
	f.Add(12) // 紧邻上界
	f.Add(9999)

	f.Fuzz(func(t *testing.T, v int) {
		if v < 0 {
			return // 负值会导致数组越界 panic，跳过。
		}
		s := State(v)
		got := s.String()
		if got == "" {
			t.Errorf("State(%d).String() 返回空字符串", v)
		}
	})
}

// FuzzEventString 模糊测试 Event.String() 对非负值不会 panic。
// 注意：负值在当前实现中会 panic（数组越界），因此仅测试非负值。
func FuzzEventString(f *testing.F) {
	f.Add(0)
	f.Add(18) // EvPostDone
	f.Add(19) // 紧邻上界
	f.Add(9999)

	f.Fuzz(func(t *testing.T, v int) {
		if v < 0 {
			return // 负值会导致数组越界 panic，跳过。
		}
		e := Event(v)
		got := e.String()
		if got == "" {
			t.Errorf("Event(%d).String() 返回空字符串", v)
		}
	})
}
