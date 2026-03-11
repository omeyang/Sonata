package mediafsm

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

// TestNewFSM 验证不同转换表创建 FSM 的行为。
func TestNewFSM(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		initial     State
		transitions []Transition
		wantState   State
	}{
		{
			"空转换表",
			Idle,
			nil,
			Idle,
		},
		{
			"单条转换",
			Idle,
			[]Transition{{Idle, EvDial, Dialing}},
			Idle,
		},
		{
			"电话场景完整表",
			Idle,
			PhoneTransitions(),
			Idle,
		},
		{
			"APP场景完整表",
			Idle,
			AppTransitions(),
			Idle,
		},
		{
			"非Idle初始状态",
			BotSpeaking,
			AppTransitions(),
			BotSpeaking,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(tt.initial, tt.transitions)
			if fsm == nil {
				t.Fatal("NewFSM 返回 nil")
			}
			if got := fsm.State(); got != tt.wantState {
				t.Errorf("初始状态 = %s, want %s", got, tt.wantState)
			}
		})
	}
}

// TestHandleValidTransition 验证有效事件能正确转换状态。
func TestHandleValidTransition(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		initial   State
		event     Event
		wantState State
	}{
		{"Idle→Dialing", Idle, EvDial, Dialing},
		{"Dialing→Ringing", Dialing, EvRinging, Ringing},
		{"Dialing→Hangup(失败)", Dialing, EvDialFailed, Hangup},
		{"Dialing→AMD(应答)", Dialing, EvAnswer, AMDDetecting},
		{"Ringing→AMD", Ringing, EvAnswer, AMDDetecting},
		{"Ringing→Hangup(超时)", Ringing, EvRingTimeout, Hangup},
		{"AMD→BotSpeaking(真人)", AMDDetecting, EvAMDHuman, BotSpeaking},
		{"AMD→Hangup(机器)", AMDDetecting, EvAMDMachine, Hangup},
		{"BotSpeaking→WaitingUser", BotSpeaking, EvBotDone, WaitingUser},
		{"BotSpeaking→BargeIn", BotSpeaking, EvBargeIn, BargeIn},
		{"WaitingUser→UserSpeaking", WaitingUser, EvSpeechStart, UserSpeaking},
		{"WaitingUser→SilenceTimeout", WaitingUser, EvSilenceTimeout, SilenceTimeout},
		{"UserSpeaking→Processing", UserSpeaking, EvSpeechEnd, Processing},
		{"Processing→BotSpeaking", Processing, EvProcessingDone, BotSpeaking},
		{"Processing→Hangup(超时)", Processing, EvProcessingTimeout, Hangup},
		{"BargeIn→UserSpeaking", BargeIn, EvBargeInDone, UserSpeaking},
		{"SilenceTimeout→BotSpeaking", SilenceTimeout, EvSilencePromptDone, BotSpeaking},
		{"SilenceTimeout→Hangup(二次)", SilenceTimeout, EvSecondSilence, Hangup},
		{"Hangup→PostProcessing", Hangup, EvPostDone, PostProcessing},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(tt.initial, PhoneTransitions())
			if err := fsm.Handle(tt.event); err != nil {
				t.Fatalf("Handle(%s) 意外出错: %v", tt.event, err)
			}
			if got := fsm.State(); got != tt.wantState {
				t.Errorf("转换后状态 = %s, want %s", got, tt.wantState)
			}
		})
	}
}

// TestHandleInvalidTransition 验证无效事件返回正确错误。
func TestHandleInvalidTransition(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		initial State
		event   Event
	}{
		{"Idle不能处理BotDone", Idle, EvBotDone},
		{"BotSpeaking不能处理Dial", BotSpeaking, EvDial},
		{"Hangup不能处理SpeechStart", Hangup, EvSpeechStart},
		{"PostProcessing不能处理任何事件", PostProcessing, EvHangup},
		{"WaitingUser不能处理BotDone", WaitingUser, EvBotDone},
		{"UserSpeaking不能处理BargeIn", UserSpeaking, EvBargeIn},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(tt.initial, PhoneTransitions())
			err := fsm.Handle(tt.event)
			if err == nil {
				t.Fatal("Handle() 应该返回错误但没有")
			}
			// 验证错误包装了 ErrInvalidTransition。
			if !errors.Is(err, ErrInvalidTransition) {
				t.Errorf("错误应该包装 ErrInvalidTransition，实际: %v", err)
			}
			// 验证错误信息包含状态和事件名。
			errMsg := err.Error()
			if got := tt.initial.String(); !containsString(errMsg, got) {
				t.Errorf("错误信息应包含状态名 %q: %s", got, errMsg)
			}
			if got := tt.event.String(); !containsString(errMsg, got) {
				t.Errorf("错误信息应包含事件名 %q: %s", got, errMsg)
			}
		})
	}
}

// TestHandleInvalidTransitionKeepsState 验证无效转换不改变状态。
func TestHandleInvalidTransitionKeepsState(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	_ = fsm.Handle(EvBotDone) // 无效事件
	if got := fsm.State(); got != Idle {
		t.Errorf("无效转换后状态应保持 Idle，实际: %s", got)
	}
}

// TestCanHandle 验证 CanHandle 在有效和无效情况下的返回值。
func TestCanHandle(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		initial State
		event   Event
		want    bool
	}{
		{"Idle可处理Dial", Idle, EvDial, true},
		{"Idle不可处理BotDone", Idle, EvBotDone, false},
		{"BotSpeaking可处理BargeIn", BotSpeaking, EvBargeIn, true},
		{"BotSpeaking可处理Hangup", BotSpeaking, EvHangup, true},
		{"BotSpeaking不可处理Dial", BotSpeaking, EvDial, false},
		{"Hangup可处理PostDone", Hangup, EvPostDone, true},
		{"Hangup不可处理Dial", Hangup, EvDial, false},
		{"PostProcessing不可处理任何", PostProcessing, EvHangup, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(tt.initial, PhoneTransitions())
			if got := fsm.CanHandle(tt.event); got != tt.want {
				t.Errorf("CanHandle(%s) = %v, want %v", tt.event, got, tt.want)
			}
		})
	}
}

// TestIsTerminal 验证终态检测。
func TestIsTerminal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		state State
		want  bool
	}{
		{"Hangup是终态", Hangup, true},
		{"PostProcessing是终态", PostProcessing, true},
		{"Idle不是终态", Idle, false},
		{"BotSpeaking不是终态", BotSpeaking, false},
		{"WaitingUser不是终态", WaitingUser, false},
		{"UserSpeaking不是终态", UserSpeaking, false},
		{"Processing不是终态", Processing, false},
		{"BargeIn不是终态", BargeIn, false},
		{"SilenceTimeout不是终态", SilenceTimeout, false},
		{"Dialing不是终态", Dialing, false},
		{"Ringing不是终态", Ringing, false},
		{"AMDDetecting不是终态", AMDDetecting, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(tt.state, PhoneTransitions())
			if got := fsm.IsTerminal(); got != tt.want {
				t.Errorf("IsTerminal() at %s = %v, want %v", tt.state, got, tt.want)
			}
		})
	}
}

// TestOnTransitionCallback 验证回调在成功转换时被正确调用。
func TestOnTransitionCallback(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())

	var (
		gotFrom  State
		gotTo    State
		gotEvent Event
		called   bool
	)

	fsm.OnTransition(func(from, to State, event Event) {
		called = true
		gotFrom = from
		gotTo = to
		gotEvent = event
	})

	if err := fsm.Handle(EvDial); err != nil {
		t.Fatalf("Handle(EvDial) 出错: %v", err)
	}

	if !called {
		t.Fatal("回调未被调用")
	}
	if gotFrom != Idle {
		t.Errorf("回调 from = %s, want Idle", gotFrom)
	}
	if gotTo != Dialing {
		t.Errorf("回调 to = %s, want Dialing", gotTo)
	}
	if gotEvent != EvDial {
		t.Errorf("回调 event = %s, want EvDial", gotEvent)
	}
}

// TestOnTransitionCallbackNotCalledOnError 验证无效转换不触发回调。
func TestOnTransitionCallbackNotCalledOnError(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())

	called := false
	fsm.OnTransition(func(from, to State, event Event) {
		called = true
	})

	_ = fsm.Handle(EvBotDone) // 无效事件

	if called {
		t.Error("无效转换不应触发回调")
	}
}

// TestOnTransitionCallbackNil 验证无回调时不会 panic。
func TestOnTransitionCallbackNil(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	// 不设置回调，直接触发转换，不应 panic。
	if err := fsm.Handle(EvDial); err != nil {
		t.Fatalf("Handle(EvDial) 出错: %v", err)
	}
}

// TestOnTransitionReplace 验证可以替换回调。
func TestOnTransitionReplace(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())

	firstCalled := false
	fsm.OnTransition(func(_, _ State, _ Event) {
		firstCalled = true
	})

	secondCalled := false
	fsm.OnTransition(func(_, _ State, _ Event) {
		secondCalled = true
	})

	if err := fsm.Handle(EvDial); err != nil {
		t.Fatalf("Handle(EvDial) 出错: %v", err)
	}

	if firstCalled {
		t.Error("旧回调不应被调用")
	}
	if !secondCalled {
		t.Error("新回调应被调用")
	}
}

// TestPhoneTransitionsFullFlow 测试电话场景的完整正常流程。
func TestPhoneTransitionsFullFlow(t *testing.T) {
	t.Parallel()

	// 完整流程: Idle→Dialing→Ringing→AMD→BotSpeaking→WaitingUser→
	// UserSpeaking→Processing→BotSpeaking→WaitingUser→SilenceTimeout→
	// BotSpeaking(提示)→WaitingUser→SilenceTimeout→Hangup→PostProcessing
	steps := []struct {
		event     Event
		wantState State
	}{
		{EvDial, Dialing},
		{EvRinging, Ringing},
		{EvAnswer, AMDDetecting},
		{EvAMDHuman, BotSpeaking},
		{EvBotDone, WaitingUser},
		{EvSpeechStart, UserSpeaking},
		{EvSpeechEnd, Processing},
		{EvProcessingDone, BotSpeaking},
		{EvBotDone, WaitingUser},
		{EvSilenceTimeout, SilenceTimeout},
		{EvSilencePromptDone, BotSpeaking},
		{EvBotDone, WaitingUser},
		{EvSilenceTimeout, SilenceTimeout},
		{EvSecondSilence, Hangup},
		{EvPostDone, PostProcessing},
	}

	fsm := NewFSM(Idle, PhoneTransitions())
	for i, step := range steps {
		if err := fsm.Handle(step.event); err != nil {
			t.Fatalf("步骤 %d Handle(%s) 出错: %v", i, step.event, err)
		}
		if got := fsm.State(); got != step.wantState {
			t.Fatalf("步骤 %d 状态 = %s, want %s", i, got, step.wantState)
		}
	}

	if !fsm.IsTerminal() {
		t.Error("流程结束后应处于终态")
	}
}

// TestPhoneTransitionsBargeInFlow 测试电话场景的打断流程。
func TestPhoneTransitionsBargeInFlow(t *testing.T) {
	t.Parallel()

	steps := []struct {
		event     Event
		wantState State
	}{
		{EvDial, Dialing},
		{EvAnswer, AMDDetecting}, // 直接应答跳过振铃
		{EvAMDHuman, BotSpeaking},
		{EvBargeIn, BargeIn},          // 用户打断
		{EvBargeInDone, UserSpeaking}, // 打断处理完成
		{EvSpeechEnd, Processing},
		{EvProcessingDone, BotSpeaking},
		{EvBotDone, WaitingUser},
		{EvHangup, Hangup},
	}

	fsm := NewFSM(Idle, PhoneTransitions())
	for i, step := range steps {
		if err := fsm.Handle(step.event); err != nil {
			t.Fatalf("步骤 %d Handle(%s) 出错: %v", i, step.event, err)
		}
		if got := fsm.State(); got != step.wantState {
			t.Fatalf("步骤 %d 状态 = %s, want %s", i, got, step.wantState)
		}
	}
}

// TestPhoneTransitionsDialFailed 测试拨号失败的短流程。
func TestPhoneTransitionsDialFailed(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	if err := fsm.Handle(EvDial); err != nil {
		t.Fatal(err)
	}
	if err := fsm.Handle(EvDialFailed); err != nil {
		t.Fatal(err)
	}
	if got := fsm.State(); got != Hangup {
		t.Errorf("拨号失败后应为 Hangup，实际: %s", got)
	}
}

// TestPhoneTransitionsAMDMachine 测试语音信箱检测流程。
func TestPhoneTransitionsAMDMachine(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	for _, ev := range []Event{EvDial, EvRinging, EvAnswer, EvAMDMachine} {
		if err := fsm.Handle(ev); err != nil {
			t.Fatalf("Handle(%s) 出错: %v", ev, err)
		}
	}
	if got := fsm.State(); got != Hangup {
		t.Errorf("检测到机器后应为 Hangup，实际: %s", got)
	}
}

// TestPhoneTransitionsHangupFromAllStates 验证所有带 Hangup 转换的状态。
func TestPhoneTransitionsHangupFromAllStates(t *testing.T) {
	t.Parallel()

	hangupStates := []State{
		Ringing, AMDDetecting, BotSpeaking, WaitingUser,
		UserSpeaking, Processing, BargeIn, SilenceTimeout,
	}

	for _, s := range hangupStates {
		t.Run(s.String(), func(t *testing.T) {
			t.Parallel()
			fsm := NewFSM(s, PhoneTransitions())
			if err := fsm.Handle(EvHangup); err != nil {
				t.Errorf("从 %s 触发 Hangup 出错: %v", s, err)
			}
			if got := fsm.State(); got != Hangup {
				t.Errorf("Hangup后状态 = %s, want Hangup", got)
			}
		})
	}
}

// TestAppTransitionsFullFlow 测试 APP 场景的完整正常流程。
func TestAppTransitionsFullFlow(t *testing.T) {
	t.Parallel()

	steps := []struct {
		event     Event
		wantState State
	}{
		{EvAnswer, BotSpeaking},  // APP 直接进入对话
		{EvBotDone, WaitingUser}, // 开场白播完
		{EvSpeechStart, UserSpeaking},
		{EvSpeechEnd, Processing},
		{EvProcessingDone, BotSpeaking},
		{EvBargeIn, BargeIn},          // 打断
		{EvBargeInDone, UserSpeaking}, // 恢复
		{EvHangup, Hangup},
		{EvPostDone, PostProcessing},
	}

	fsm := NewFSM(Idle, AppTransitions())
	for i, step := range steps {
		if err := fsm.Handle(step.event); err != nil {
			t.Fatalf("步骤 %d Handle(%s) 出错: %v", i, step.event, err)
		}
		if got := fsm.State(); got != step.wantState {
			t.Fatalf("步骤 %d 状态 = %s, want %s", i, got, step.wantState)
		}
	}
}

// TestAppTransitionsNoDial 验证 APP 场景不支持拨号事件。
func TestAppTransitionsNoDial(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, AppTransitions())
	if fsm.CanHandle(EvDial) {
		t.Error("APP 场景不应支持 EvDial")
	}
	err := fsm.Handle(EvDial)
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("APP 场景 Dial 应返回 ErrInvalidTransition，实际: %v", err)
	}
}

// TestAppTransitionsSilenceEscalation 验证 APP 场景的静默升级路径。
func TestAppTransitionsSilenceEscalation(t *testing.T) {
	t.Parallel()

	steps := []struct {
		event     Event
		wantState State
	}{
		{EvAnswer, BotSpeaking},
		{EvBotDone, WaitingUser},
		{EvSilenceTimeout, SilenceTimeout},
		{EvSilencePromptDone, BotSpeaking}, // 播放静默提示
		{EvBotDone, WaitingUser},
		{EvSilenceTimeout, SilenceTimeout},
		{EvSecondSilence, Hangup}, // 二次静默→挂断
	}

	fsm := NewFSM(Idle, AppTransitions())
	for i, step := range steps {
		if err := fsm.Handle(step.event); err != nil {
			t.Fatalf("步骤 %d Handle(%s) 出错: %v", i, step.event, err)
		}
		if got := fsm.State(); got != step.wantState {
			t.Fatalf("步骤 %d 状态 = %s, want %s", i, got, step.wantState)
		}
	}
}

// TestAppTransitionsProcessingTimeout 验证处理超时路径。
func TestAppTransitionsProcessingTimeout(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Processing, AppTransitions())
	if err := fsm.Handle(EvProcessingTimeout); err != nil {
		t.Fatalf("Handle(EvProcessingTimeout) 出错: %v", err)
	}
	if got := fsm.State(); got != Hangup {
		t.Errorf("处理超时后应为 Hangup，实际: %s", got)
	}
}

// TestConcurrentHandle 验证并发 Handle 不会产生数据竞争或 panic。
func TestConcurrentHandle(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	var callCount atomic.Int64
	fsm.OnTransition(func(_, _ State, _ Event) {
		callCount.Add(1)
	})

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for range goroutines {
		go func() {
			defer wg.Done()
			// 多个 goroutine 竞争处理事件，只有一个会成功让 Idle→Dialing。
			_ = fsm.Handle(EvDial)
		}()
	}

	wg.Wait()

	// FSM 应该处于某个有效状态（Dialing 或仍在 Idle 如果所有转换都失败了——但这不会发生）。
	got := fsm.State()
	if got != Dialing {
		t.Errorf("并发后状态应为 Dialing，实际: %s", got)
	}
}

// TestConcurrentStateAndHandle 验证并发读写不会产生数据竞争。
func TestConcurrentStateAndHandle(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(BotSpeaking, AppTransitions())
	fsm.OnTransition(func(_, _ State, _ Event) {})

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines * 4)

	for range goroutines {
		// 并发读取状态。
		go func() {
			defer wg.Done()
			_ = fsm.State()
		}()
		// 并发检查 CanHandle。
		go func() {
			defer wg.Done()
			_ = fsm.CanHandle(EvBotDone)
		}()
		// 并发检查 IsTerminal。
		go func() {
			defer wg.Done()
			_ = fsm.IsTerminal()
		}()
		// 并发 Handle。
		go func() {
			defer wg.Done()
			_ = fsm.Handle(EvBotDone)
		}()
	}

	wg.Wait()
}

// TestConcurrentOnTransition 验证并发设置回调不会产生数据竞争。
func TestConcurrentOnTransition(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for range goroutines {
		go func() {
			defer wg.Done()
			fsm.OnTransition(func(_, _ State, _ Event) {})
		}()
	}

	wg.Wait()
}

// TestErrInvalidTransitionSentinel 验证 ErrInvalidTransition 是可用的哨兵错误。
func TestErrInvalidTransitionSentinel(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	err := fsm.Handle(EvBotDone)

	// errors.Is 应该能匹配。
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("errors.Is 应匹配 ErrInvalidTransition")
	}

	// 同时错误应该不等于裸哨兵（因为有包装信息）。
	if err.Error() == ErrInvalidTransition.Error() {
		t.Error("包装后的错误信息不应与裸哨兵完全相同")
	}
}

// TestPhoneTransitionsRingTimeout 测试振铃超时流程。
func TestPhoneTransitionsRingTimeout(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, PhoneTransitions())
	for _, ev := range []Event{EvDial, EvRinging, EvRingTimeout} {
		if err := fsm.Handle(ev); err != nil {
			t.Fatalf("Handle(%s) 出错: %v", ev, err)
		}
	}
	if got := fsm.State(); got != Hangup {
		t.Errorf("振铃超时后应为 Hangup，实际: %s", got)
	}
}

// TestTransitionOverwrite 验证重复的转换键会被后者覆盖。
func TestTransitionOverwrite(t *testing.T) {
	t.Parallel()

	// 两条规则有相同的 (From, Event)，后者应覆盖前者。
	transitions := []Transition{
		{Idle, EvDial, Dialing},
		{Idle, EvDial, BotSpeaking}, // 覆盖
	}

	fsm := NewFSM(Idle, transitions)
	if err := fsm.Handle(EvDial); err != nil {
		t.Fatal(err)
	}
	if got := fsm.State(); got != BotSpeaking {
		t.Errorf("重复键应使用后者，状态 = %s, want BotSpeaking", got)
	}
}

// TestEmptyTransitions 验证空转换表下所有事件都无效。
func TestEmptyTransitions(t *testing.T) {
	t.Parallel()

	fsm := NewFSM(Idle, nil)
	err := fsm.Handle(EvDial)
	if !errors.Is(err, ErrInvalidTransition) {
		t.Errorf("空转换表应返回 ErrInvalidTransition，实际: %v", err)
	}
	if fsm.CanHandle(EvDial) {
		t.Error("空转换表不应能处理任何事件")
	}
}

// TestCallbackParameters 验证多次转换的回调参数都正确。
func TestCallbackParameters(t *testing.T) {
	t.Parallel()

	type record struct {
		from  State
		to    State
		event Event
	}
	var records []record

	fsm := NewFSM(Idle, PhoneTransitions())
	fsm.OnTransition(func(from, to State, event Event) {
		records = append(records, record{from, to, event})
	})

	steps := []struct {
		event Event
		from  State
		to    State
	}{
		{EvDial, Idle, Dialing},
		{EvRinging, Dialing, Ringing},
		{EvAnswer, Ringing, AMDDetecting},
	}

	for _, step := range steps {
		if err := fsm.Handle(step.event); err != nil {
			t.Fatal(err)
		}
	}

	if len(records) != len(steps) {
		t.Fatalf("回调次数 = %d, want %d", len(records), len(steps))
	}

	for i, step := range steps {
		r := records[i]
		if r.from != step.from || r.to != step.to || r.event != step.event {
			t.Errorf("第 %d 次回调参数不匹配: got {%s,%s,%s}, want {%s,%s,%s}",
				i, r.from, r.to, r.event, step.from, step.to, step.event)
		}
	}
}

// ---------- Unsynced 选项测试 ----------

// TestUnsyncedFSM_BasicTransitions 验证无锁 FSM 的基本转换正确性。
func TestUnsyncedFSM_BasicTransitions(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Idle, AppTransitions(), Unsynced())

	if err := fsm.Handle(EvAnswer); err != nil {
		t.Fatalf("Handle(EvAnswer): %v", err)
	}
	if fsm.State() != BotSpeaking {
		t.Fatalf("state = %s, want BOT_SPEAKING", fsm.State())
	}

	if err := fsm.Handle(EvBotDone); err != nil {
		t.Fatalf("Handle(EvBotDone): %v", err)
	}
	if fsm.State() != WaitingUser {
		t.Fatalf("state = %s, want WAITING_USER", fsm.State())
	}

	if err := fsm.Handle(EvSpeechStart); err != nil {
		t.Fatalf("Handle(EvSpeechStart): %v", err)
	}
	if fsm.State() != UserSpeaking {
		t.Fatalf("state = %s, want USER_SPEAKING", fsm.State())
	}
}

// TestUnsyncedFSM_InvalidTransition 验证无锁 FSM 拒绝无效转换。
func TestUnsyncedFSM_InvalidTransition(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Idle, AppTransitions(), Unsynced())

	err := fsm.Handle(EvBotDone)
	if err == nil {
		t.Fatal("expected error for invalid transition")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("error = %v, want ErrInvalidTransition", err)
	}
	if fsm.State() != Idle {
		t.Fatalf("state changed on error: %s", fsm.State())
	}
}

// TestUnsyncedFSM_CanHandle 验证无锁 FSM 的 CanHandle。
func TestUnsyncedFSM_CanHandle(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Idle, AppTransitions(), Unsynced())

	if !fsm.CanHandle(EvAnswer) {
		t.Error("should handle EvAnswer from Idle")
	}
	if fsm.CanHandle(EvBotDone) {
		t.Error("should not handle EvBotDone from Idle")
	}
}

// TestUnsyncedFSM_IsTerminal 验证无锁 FSM 的终态检测。
func TestUnsyncedFSM_IsTerminal(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Hangup, AppTransitions(), Unsynced())

	if !fsm.IsTerminal() {
		t.Error("Hangup should be terminal")
	}
}

// TestUnsyncedFSM_Callback 验证无锁 FSM 的回调正确触发。
func TestUnsyncedFSM_Callback(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Idle, AppTransitions(), Unsynced())

	var called bool
	var cbFrom, cbTo State
	var cbEvent Event
	fsm.OnTransition(func(from, to State, event Event) {
		called = true
		cbFrom = from
		cbTo = to
		cbEvent = event
	})

	if err := fsm.Handle(EvAnswer); err != nil {
		t.Fatalf("Handle(EvAnswer): %v", err)
	}
	if !called {
		t.Fatal("callback not called")
	}
	if cbFrom != Idle || cbTo != BotSpeaking || cbEvent != EvAnswer {
		t.Errorf("callback args = {%s,%s,%s}, want {IDLE,BOT_SPEAKING,ANSWER}", cbFrom, cbTo, cbEvent)
	}
}

// TestDefaultFSM_BackwardCompat 验证默认 FSM 向后兼容（无选项时使用互斥锁）。
func TestDefaultFSM_BackwardCompat(t *testing.T) {
	t.Parallel()
	fsm := NewFSM(Idle, AppTransitions())

	if err := fsm.Handle(EvAnswer); err != nil {
		t.Fatalf("Handle(EvAnswer): %v", err)
	}
	if fsm.State() != BotSpeaking {
		t.Fatalf("state = %s, want BOT_SPEAKING", fsm.State())
	}
}

// BenchmarkHandle 基准测试 Handle 热路径（成功转换）。
func BenchmarkHandle(b *testing.B) {
	transitions := PhoneTransitions()
	// 反复在 BotSpeaking↔WaitingUser 之间切换。
	fsm := NewFSM(BotSpeaking, transitions)

	b.ResetTimer()
	for i := range b.N {
		if i%2 == 0 {
			_ = fsm.Handle(EvBotDone)
		} else {
			_ = fsm.Handle(EvSpeechStart)
		}
	}
}

// BenchmarkHandleInvalid 基准测试 Handle 热路径（失败转换）。
func BenchmarkHandleInvalid(b *testing.B) {
	fsm := NewFSM(Idle, PhoneTransitions())

	b.ResetTimer()
	for range b.N {
		_ = fsm.Handle(EvBotDone)
	}
}

// BenchmarkCanHandle 基准测试 CanHandle 查找。
func BenchmarkCanHandle(b *testing.B) {
	fsm := NewFSM(BotSpeaking, PhoneTransitions())

	b.ResetTimer()
	for range b.N {
		_ = fsm.CanHandle(EvBargeIn)
	}
}

// BenchmarkState 基准测试 State 读取。
func BenchmarkState(b *testing.B) {
	fsm := NewFSM(BotSpeaking, PhoneTransitions())

	b.ResetTimer()
	for range b.N {
		_ = fsm.State()
	}
}

// containsString 检查 s 是否包含 substr。
func containsString(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
