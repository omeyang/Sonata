package mediafsm

import (
	"errors"
	"fmt"
	"sync"
)

// rwLocker 抽象读写锁行为，允许在不同同步策略间切换。
type rwLocker interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()
}

// nopRWLocker 无操作的读写锁，供单 goroutine 场景使用。
type nopRWLocker struct{}

func (nopRWLocker) Lock()    {}
func (nopRWLocker) Unlock()  {}
func (nopRWLocker) RLock()   {}
func (nopRWLocker) RUnlock() {}

// Option 配置 FSM 行为。
type Option func(*FSM)

// ErrInvalidTransition 在当前状态无法处理事件时返回。
var ErrInvalidTransition = errors.New("invalid state transition")

// Callback 在状态转换成功后被调用。
type Callback func(from, to State, event Event)

// transitionKey 用于 O(1) 查找。
type transitionKey struct {
	from  State
	event Event
}

// FSM 是表驱动的媒体状态机。
// 默认使用 sync.RWMutex 保证并发安全；通过 Unsynced() 选项可关闭同步，
// 仅限单 goroutine 使用（如 Session.eventLoop）。
// 转换规则由调用方通过 NewFSM 传入。
type FSM struct {
	locker   rwLocker
	state    State
	index    map[transitionKey]State
	callback Callback
}

// Unsynced 创建无同步的 FSM，消除互斥锁开销。
// 仅适用于所有 Handle/State/CanHandle/IsTerminal 调用
// 都在同一个 goroutine 中执行的场景（如 Session.eventLoop）。
func Unsynced() Option {
	return func(f *FSM) {
		f.locker = nopRWLocker{}
	}
}

// NewFSM 创建以给定状态为初始状态的新媒体 FSM。
// transitions 定义所有有效的状态转换规则。
// 默认使用 sync.RWMutex 保证并发安全，可通过 Unsynced() 关闭同步。
func NewFSM(initial State, transitions []Transition, opts ...Option) *FSM {
	index := make(map[transitionKey]State, len(transitions))
	for _, t := range transitions {
		index[transitionKey{t.From, t.Event}] = t.To
	}

	fsm := &FSM{
		locker: &sync.RWMutex{},
		state:  initial,
		index:  index,
	}
	for _, opt := range opts {
		opt(fsm)
	}

	return fsm
}

// OnTransition 设置每次成功转换后调用的回调函数。
func (f *FSM) OnTransition(cb Callback) {
	f.locker.Lock()
	defer f.locker.Unlock()
	f.callback = cb
}

// State 返回当前状态。
func (f *FSM) State() State {
	f.locker.RLock()
	defer f.locker.RUnlock()
	return f.state
}

// Handle 处理事件，若有效则转换到下一状态。
func (f *FSM) Handle(event Event) error {
	f.locker.Lock()
	defer f.locker.Unlock()

	key := transitionKey{f.state, event}
	next, ok := f.index[key]
	if !ok {
		return fmt.Errorf("%w: %s + %s", ErrInvalidTransition, f.state, event)
	}

	prev := f.state
	f.state = next

	if f.callback != nil {
		f.callback(prev, next, event)
	}

	return nil
}

// CanHandle 当事件在当前状态下有效时返回 true。
func (f *FSM) CanHandle(event Event) bool {
	f.locker.RLock()
	defer f.locker.RUnlock()
	_, ok := f.index[transitionKey{f.state, event}]
	return ok
}

// IsTerminal 当 FSM 处于终态（Hangup 或 PostProcessing）时返回 true。
func (f *FSM) IsTerminal() bool {
	f.locker.RLock()
	defer f.locker.RUnlock()
	return f.state == Hangup || f.state == PostProcessing
}

// PhoneTransitions 返回电话场景的完整状态转换表。
// 包含拨号、振铃、AMD 等电话特有的前置状态。
func PhoneTransitions() []Transition {
	return []Transition{
		// 拨号流程。
		{Idle, EvDial, Dialing},
		{Dialing, EvRinging, Ringing},
		{Dialing, EvDialFailed, Hangup},
		{Dialing, EvAnswer, AMDDetecting},

		// 振铃流程。
		{Ringing, EvAnswer, AMDDetecting},
		{Ringing, EvRingTimeout, Hangup},
		{Ringing, EvHangup, Hangup},

		// AMD 流程。
		{AMDDetecting, EvAMDHuman, BotSpeaking},
		{AMDDetecting, EvAMDMachine, Hangup},
		{AMDDetecting, EvHangup, Hangup},

		// 以下为通用对话流程（与 AppTransitions 共享）。
		{BotSpeaking, EvBotDone, WaitingUser},
		{BotSpeaking, EvBargeIn, BargeIn},
		{BotSpeaking, EvHangup, Hangup},

		{WaitingUser, EvSpeechStart, UserSpeaking},
		{WaitingUser, EvSilenceTimeout, SilenceTimeout},
		{WaitingUser, EvHangup, Hangup},

		{UserSpeaking, EvSpeechEnd, Processing},
		{UserSpeaking, EvHangup, Hangup},

		{Processing, EvProcessingDone, BotSpeaking},
		{Processing, EvProcessingTimeout, Hangup},
		{Processing, EvHangup, Hangup},

		{BargeIn, EvBargeInDone, UserSpeaking},
		{BargeIn, EvHangup, Hangup},

		{SilenceTimeout, EvSilencePromptDone, BotSpeaking},
		{SilenceTimeout, EvSecondSilence, Hangup},
		{SilenceTimeout, EvHangup, Hangup},

		{Hangup, EvPostDone, PostProcessing},
	}
}

// AppTransitions 返回 APP 场景的状态转换表。
// 无拨号/振铃/AMD，会话从 BotSpeaking（开场白）直接开始。
func AppTransitions() []Transition {
	return []Transition{
		// APP 会话建立：Idle → BotSpeaking（播放开场白）。
		{Idle, EvAnswer, BotSpeaking},

		// 通用对话流程。
		{BotSpeaking, EvBotDone, WaitingUser},
		{BotSpeaking, EvBargeIn, BargeIn},
		{BotSpeaking, EvHangup, Hangup},

		{WaitingUser, EvSpeechStart, UserSpeaking},
		{WaitingUser, EvSilenceTimeout, SilenceTimeout},
		{WaitingUser, EvHangup, Hangup},

		{UserSpeaking, EvSpeechEnd, Processing},
		{UserSpeaking, EvHangup, Hangup},

		{Processing, EvProcessingDone, BotSpeaking},
		{Processing, EvProcessingTimeout, Hangup},
		{Processing, EvHangup, Hangup},

		{BargeIn, EvBargeInDone, UserSpeaking},
		{BargeIn, EvHangup, Hangup},

		{SilenceTimeout, EvSilencePromptDone, BotSpeaking},
		{SilenceTimeout, EvSecondSilence, Hangup},
		{SilenceTimeout, EvHangup, Hangup},

		{Hangup, EvPostDone, PostProcessing},
	}
}
