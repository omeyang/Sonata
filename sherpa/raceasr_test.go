package sherpa

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/omeyang/Sonata/engine/aiface"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- 配置测试 ---

func TestNewRacingASR_NilLocal(t *testing.T) {
	t.Parallel()

	_, err := NewRacingASR(RacingASRConfig{Cloud: &mockASRProvider{}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Local")
}

func TestNewRacingASR_NilCloud(t *testing.T) {
	t.Parallel()

	_, err := NewRacingASR(RacingASRConfig{Local: &mockASRProvider{}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Cloud")
}

func TestNewRacingASR_OK(t *testing.T) {
	t.Parallel()

	r, err := NewRacingASR(RacingASRConfig{
		Local: &mockASRProvider{},
		Cloud: &mockASRProvider{},
	})
	require.NoError(t, err)
	assert.NotNil(t, r)
}

// --- 竞速策略测试 ---

func TestRacingASR_LocalFinalWins(t *testing.T) {
	t.Parallel()

	localEvents := make(chan aiface.ASREvent, 4)
	cloudEvents := make(chan aiface.ASREvent, 4)

	local := &mockASRProvider{stream: &mockASRStream{events: localEvents}}
	cloud := &mockASRProvider{stream: &mockASRStream{events: cloudEvents}}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 本地先发 final。
	localEvents <- aiface.ASREvent{Text: "你好", IsFinal: true, LatencyMs: 100}

	evt := readEvent(t, stream.Events(), time.Second)
	assert.Equal(t, "你好", evt.Text)
	assert.True(t, evt.IsFinal)

	// 云端后发 final 应被丢弃。
	cloudEvents <- aiface.ASREvent{Text: "你好世界", IsFinal: true, LatencyMs: 500}

	// 关闭源通道让 mergeLoop 退出。
	close(localEvents)
	close(cloudEvents)

	// 收集剩余事件，不应包含云端 final。
	remaining := drainEvents(stream.Events(), 200*time.Millisecond)
	for _, e := range remaining {
		assert.False(t, e.IsFinal, "云端 final 应被丢弃")
	}
}

func TestRacingASR_CloudFinalWins(t *testing.T) {
	t.Parallel()

	localEvents := make(chan aiface.ASREvent, 4)
	cloudEvents := make(chan aiface.ASREvent, 4)

	local := &mockASRProvider{stream: &mockASRStream{events: localEvents}}
	cloud := &mockASRProvider{stream: &mockASRStream{events: cloudEvents}}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 云端先发 final。
	cloudEvents <- aiface.ASREvent{Text: "你好世界", IsFinal: true, LatencyMs: 300}

	evt := readEvent(t, stream.Events(), time.Second)
	assert.Equal(t, "你好世界", evt.Text)
	assert.True(t, evt.IsFinal)

	close(localEvents)
	close(cloudEvents)
}

func TestRacingASR_PartialEvents(t *testing.T) {
	t.Parallel()

	localEvents := make(chan aiface.ASREvent, 4)
	cloudEvents := make(chan aiface.ASREvent, 4)

	local := &mockASRProvider{stream: &mockASRStream{events: localEvents}}
	cloud := &mockASRProvider{stream: &mockASRStream{events: cloudEvents}}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 两个源都发 partial，都应转发。
	localEvents <- aiface.ASREvent{Text: "你", IsFinal: false}
	cloudEvents <- aiface.ASREvent{Text: "你好", IsFinal: false}

	events := drainEvents(stream.Events(), 200*time.Millisecond)
	assert.GreaterOrEqual(t, len(events), 1, "至少应收到一个 partial")

	close(localEvents)
	close(cloudEvents)
}

func TestRacingASR_FeedAudio_BothStreams(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{events: make(chan aiface.ASREvent, 4)}
	cloudStream := &mockASRStream{events: make(chan aiface.ASREvent, 4)}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	chunk := make([]byte, 320)
	err = stream.FeedAudio(context.Background(), chunk)
	assert.NoError(t, err)

	// 两个流都应收到音频。
	assert.Equal(t, 1, localStream.feedCount())
	assert.Equal(t, 1, cloudStream.feedCount())

	close(localStream.events)
	close(cloudStream.events)
}

func TestRacingASR_FeedAudio_OneStreamFails(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{
		events:  make(chan aiface.ASREvent, 4),
		feedErr: errors.New("local feed error"),
	}
	cloudStream := &mockASRStream{events: make(chan aiface.ASREvent, 4)}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 一个流失败时不应返回错误（另一个仍可用）。
	err = stream.FeedAudio(context.Background(), make([]byte, 320))
	assert.NoError(t, err)

	close(localStream.events)
	close(cloudStream.events)
}

func TestRacingASR_FeedAudio_BothFail(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{
		events:  make(chan aiface.ASREvent, 4),
		feedErr: errors.New("local fail"),
	}
	cloudStream := &mockASRStream{
		events:  make(chan aiface.ASREvent, 4),
		feedErr: errors.New("cloud fail"),
	}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 两个都失败应返回错误。
	err = stream.FeedAudio(context.Background(), make([]byte, 320))
	assert.Error(t, err)

	close(localStream.events)
	close(cloudStream.events)
}

func TestRacingASR_Close_Idempotent(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{events: make(chan aiface.ASREvent, 1)}
	cloudStream := &mockASRStream{events: make(chan aiface.ASREvent, 1)}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)

	close(localStream.events)
	close(cloudStream.events)

	assert.NoError(t, stream.Close())
	assert.NoError(t, stream.Close())
}

func TestRacingASR_FeedAudio_AfterClose(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{events: make(chan aiface.ASREvent, 1)}
	cloudStream := &mockASRStream{events: make(chan aiface.ASREvent, 1)}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)

	close(localStream.events)
	close(cloudStream.events)

	stream.Close()

	err = stream.FeedAudio(context.Background(), make([]byte, 320))
	assert.Error(t, err)
}

// --- StartStream 错误路径 ---

func TestRacingASR_StartStream_LocalFails(t *testing.T) {
	t.Parallel()

	local := &mockASRProvider{startErr: errors.New("local init error")}
	cloud := &mockASRProvider{stream: &mockASRStream{events: make(chan aiface.ASREvent, 1)}}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	_, err = r.StartStream(context.Background(), aiface.ASRConfig{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "本地 ASR")
}

func TestRacingASR_StartStream_CloudFails(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{events: make(chan aiface.ASREvent, 1)}
	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{startErr: errors.New("cloud init error")}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	_, err = r.StartStream(context.Background(), aiface.ASRConfig{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "云端 ASR")
	// 验证本地流被关闭了。
	localStream.mu.Lock()
	assert.True(t, localStream.closed)
	localStream.mu.Unlock()
}

func TestRacingASR_FeedAudio_CloudOnlyFails(t *testing.T) {
	t.Parallel()

	localStream := &mockASRStream{events: make(chan aiface.ASREvent, 4)}
	cloudStream := &mockASRStream{
		events:  make(chan aiface.ASREvent, 4),
		feedErr: errors.New("cloud feed error"),
	}

	local := &mockASRProvider{stream: localStream}
	cloud := &mockASRProvider{stream: cloudStream}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 仅云端失败不应返回错误。
	err = stream.FeedAudio(context.Background(), make([]byte, 320))
	assert.NoError(t, err)

	close(localStream.events)
	close(cloudStream.events)
}

func TestRacingASR_MergeLoop_LocalChannelCloses(t *testing.T) {
	t.Parallel()

	localEvents := make(chan aiface.ASREvent, 4)
	cloudEvents := make(chan aiface.ASREvent, 4)

	local := &mockASRProvider{stream: &mockASRStream{events: localEvents}}
	cloud := &mockASRProvider{stream: &mockASRStream{events: cloudEvents}}

	r, err := NewRacingASR(RacingASRConfig{Local: local, Cloud: cloud})
	require.NoError(t, err)

	stream, err := r.StartStream(context.Background(), aiface.ASRConfig{})
	require.NoError(t, err)
	defer stream.Close()

	// 关闭本地通道，云端仍然发送 final。
	close(localEvents)
	cloudEvents <- aiface.ASREvent{Text: "云端结果", IsFinal: true, LatencyMs: 200}
	close(cloudEvents)

	events := drainEvents(stream.Events(), 500*time.Millisecond)
	hasFinal := false
	for _, e := range events {
		if e.IsFinal && e.Text == "云端结果" {
			hasFinal = true
		}
	}
	assert.True(t, hasFinal, "关闭本地通道后云端 final 应被转发")
}

// --- Mock 实现 ---

// mockASRProvider 模拟 ASR 提供者。
type mockASRProvider struct {
	stream   *mockASRStream
	startErr error
}

func (m *mockASRProvider) StartStream(_ context.Context, _ aiface.ASRConfig) (aiface.ASRStream, error) {
	if m.startErr != nil {
		return nil, m.startErr
	}
	return m.stream, nil
}

// mockASRStream 模拟 ASR 流。
type mockASRStream struct {
	events  chan aiface.ASREvent
	feedErr error

	mu     sync.Mutex
	feeds  int
	closed bool
}

func (m *mockASRStream) FeedAudio(_ context.Context, _ []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.feeds++
	return m.feedErr
}

func (m *mockASRStream) Events() <-chan aiface.ASREvent {
	return m.events
}

func (m *mockASRStream) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func (m *mockASRStream) feedCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.feeds
}

// --- 辅助函数 ---

// readEvent 从通道读取一个事件，超时则失败。
func readEvent(t *testing.T, ch <-chan aiface.ASREvent, timeout time.Duration) aiface.ASREvent {
	t.Helper()
	select {
	case evt := <-ch:
		return evt
	case <-time.After(timeout):
		t.Fatal("读取事件超时")
		return aiface.ASREvent{}
	}
}

// drainEvents 在指定时间内收集通道中的所有事件。
func drainEvents(ch <-chan aiface.ASREvent, timeout time.Duration) []aiface.ASREvent {
	var events []aiface.ASREvent
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case evt, ok := <-ch:
			if !ok {
				return events
			}
			events = append(events, evt)
		case <-timer.C:
			return events
		}
	}
}
