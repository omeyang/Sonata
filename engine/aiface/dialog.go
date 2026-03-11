package aiface

import "context"

// DialogEngine 是业务对话引擎的接口。
// Session 运行时在 ASR 返回最终结果后调用此接口生成回复。
//
// 不同产品实现不同的对话逻辑：
//   - 外呼产品：状态机 + 规则引擎（开场→资质→收集→异议→结束）
//   - 外语学习：课程引擎（课程→练习→纠正→鼓励→总结）
//   - 故事机：故事引擎（选择→讲述→互动→结尾）
type DialogEngine interface {
	// Opening 返回对话开始时的第一句话。
	Opening() string

	// Process 处理用户输入，返回回复文本。
	// 非流式模式，适用于简单场景或 LLM 不可用时的降级。
	Process(ctx context.Context, userText string) (string, error)

	// ProcessStream 流式处理用户输入，按句返回回复文本。
	// 通道关闭表示所有句子已发送。
	ProcessStream(ctx context.Context, userText string) (<-chan string, error)

	// Finished 当对话已结束时返回 true。
	Finished() bool
}
