// Package mediafsm 实现通用的媒体状态机（MSM）。
//
// MSM 管理实时语音交互的生命周期，控制"何时听、何时说"。
// 它与业务对话引擎并行运行：MSM 管节奏，对话引擎管内容。
//
// 状态机采用表驱动设计，转换规则由调用方通过 Transition 切片提供，
// 不同产品可定义不同的状态流转（如电话场景有拨号/振铃，APP 场景没有）。
package mediafsm

// State 表示媒体状态机中的状态。
type State int

// 通用媒体状态。
// 产品可以只使用其中的子集（如 APP 不需要 Dialing/Ringing/AMDDetecting）。
const (
	Idle           State = iota // 等待开始。
	Dialing                     // 正在拨号（电话场景）。
	Ringing                     // 对方振铃中（电话场景）。
	AMDDetecting                // 语音信箱检测（电话场景）。
	BotSpeaking                 // AI 正在播放语音。
	WaitingUser                 // 等待用户说话。
	UserSpeaking                // 用户正在说话，ASR 激活。
	Processing                  // ASR 返回结果，LLM+规则运行中。
	BargeIn                     // 用户打断了 AI 播放。
	SilenceTimeout              // 超时未检测到语音。
	Hangup                      // 会话结束。
	PostProcessing              // 异步后处理。
)

func (s State) String() string {
	names := [...]string{
		"IDLE", "DIALING", "RINGING", "AMD_DETECTING",
		"BOT_SPEAKING", "WAITING_USER", "USER_SPEAKING", "PROCESSING",
		"BARGE_IN", "SILENCE_TIMEOUT", "HANGUP", "POST_PROCESSING",
	}
	if int(s) >= 0 && int(s) < len(names) {
		return names[s]
	}
	return "UNKNOWN"
}

// Event 触发媒体状态机中的状态转换。
type Event int

// 通用媒体事件。
const (
	EvDial              Event = iota // 发起呼叫/会话。
	EvRinging                        // 对方振铃。
	EvDialFailed                     // 呼叫失败。
	EvAnswer                         // 对方应答/会话建立。
	EvRingTimeout                    // 振铃超时。
	EvAMDHuman                       // 检测到真人。
	EvAMDMachine                     // 检测到语音信箱。
	EvBotDone                        // AI 播放完成。
	EvSpeechStart                    // VAD 检测到语音开始。
	EvSpeechEnd                      // ASR 返回最终结果。
	EvBargeIn                        // 用户打断。
	EvBargeInDone                    // 打断处理完成。
	EvProcessingDone                 // LLM+TTS 首个分片就绪。
	EvSilenceTimeout                 // 静默超过阈值。
	EvSilencePromptDone              // 静默提示已播放。
	EvSecondSilence                  // 第二次静默超时。
	EvHangup                         // 会话结束。
	EvProcessingTimeout              // 处理超时。
	EvPostDone                       // 后处理完成。
)

func (e Event) String() string {
	names := [...]string{
		"DIAL", "RINGING", "DIAL_FAILED", "ANSWER", "RING_TIMEOUT",
		"AMD_HUMAN", "AMD_MACHINE", "BOT_DONE", "SPEECH_START", "SPEECH_END",
		"BARGE_IN", "BARGE_IN_DONE", "PROCESSING_DONE", "SILENCE_TIMEOUT",
		"SILENCE_PROMPT_DONE", "SECOND_SILENCE", "HANGUP", "PROCESSING_TIMEOUT",
		"POST_DONE",
	}
	if int(e) >= 0 && int(e) < len(names) {
		return names[e]
	}
	return "UNKNOWN"
}

// Transition 定义单个有效的 (源状态, 事件) → 目标状态 映射。
type Transition struct {
	From  State
	Event Event
	To    State
}
