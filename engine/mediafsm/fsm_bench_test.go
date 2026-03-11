package mediafsm

import "testing"

// BenchmarkHandleConcurrent 基准测试并发 FSM 转换（多 session 共享转换表场景）。
func BenchmarkHandleConcurrent(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		// 每个 goroutine 创建独立 FSM 实例（模拟独立 session）。
		fsm := NewFSM(Idle, AppTransitions())
		_ = fsm.Handle(EvAnswer)
		_ = fsm.Handle(EvBotDone)

		for pb.Next() {
			// WaitingUser → UserSpeaking → Processing → BotSpeaking → WaitingUser 循环。
			_ = fsm.Handle(EvSpeechStart)
			_ = fsm.Handle(EvSpeechEnd)
			_ = fsm.Handle(EvProcessingDone)
			_ = fsm.Handle(EvBotDone)
		}
	})
}

// BenchmarkNewFSM 基准测试 FSM 创建（session 初始化路径）。
func BenchmarkNewFSM(b *testing.B) {
	transitions := AppTransitions()
	b.ResetTimer()
	for range b.N {
		_ = NewFSM(Idle, transitions)
	}
}

// BenchmarkFullConversationCycle 基准测试完整对话周期的 FSM 转换（synced）。
func BenchmarkFullConversationCycle(b *testing.B) {
	fsm := NewFSM(Idle, AppTransitions())
	_ = fsm.Handle(EvAnswer)
	_ = fsm.Handle(EvBotDone)

	b.ResetTimer()
	for range b.N {
		_ = fsm.Handle(EvSpeechStart)
		_ = fsm.Handle(EvSpeechEnd)
		_ = fsm.Handle(EvProcessingDone)
		_ = fsm.Handle(EvBotDone)
	}
}

// BenchmarkFullConversationCycleUnsynced 无锁版本的完整对话周期。
func BenchmarkFullConversationCycleUnsynced(b *testing.B) {
	fsm := NewFSM(Idle, AppTransitions(), Unsynced())
	_ = fsm.Handle(EvAnswer)
	_ = fsm.Handle(EvBotDone)

	b.ResetTimer()
	for range b.N {
		_ = fsm.Handle(EvSpeechStart)
		_ = fsm.Handle(EvSpeechEnd)
		_ = fsm.Handle(EvProcessingDone)
		_ = fsm.Handle(EvBotDone)
	}
}

// BenchmarkHandleUnsynced 无锁版本的单次 Handle。
func BenchmarkHandleUnsynced(b *testing.B) {
	fsm := NewFSM(BotSpeaking, PhoneTransitions(), Unsynced())

	b.ResetTimer()
	for i := range b.N {
		if i%2 == 0 {
			_ = fsm.Handle(EvBotDone)
		} else {
			_ = fsm.Handle(EvSpeechStart)
		}
	}
}

// BenchmarkStateUnsynced 无锁版本的 State 读取。
func BenchmarkStateUnsynced(b *testing.B) {
	fsm := NewFSM(BotSpeaking, PhoneTransitions(), Unsynced())

	b.ResetTimer()
	for range b.N {
		_ = fsm.State()
	}
}
