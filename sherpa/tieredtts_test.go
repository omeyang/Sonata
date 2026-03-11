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

func TestNewTieredTTS_NilLocal(t *testing.T) {
	t.Parallel()

	_, err := NewTieredTTS(TieredTTSConfig{Cloud: &mockTTSProvider{}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Local")
}

func TestNewTieredTTS_NilCloud(t *testing.T) {
	t.Parallel()

	_, err := NewTieredTTS(TieredTTSConfig{Local: &LocalTTS{}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Cloud")
}

func TestTieredTTSConfig_Defaults(t *testing.T) {
	t.Parallel()

	cfg := TieredTTSConfig{
		Local: &LocalTTS{},
		Cloud: &mockTTSProvider{},
	}
	cfg.setDefaults()

	assert.Equal(t, 10, cfg.Threshold)
}

func TestTieredTTSConfig_CustomThreshold(t *testing.T) {
	t.Parallel()

	cfg := TieredTTSConfig{
		Local:     &LocalTTS{},
		Cloud:     &mockTTSProvider{},
		Threshold: 5,
	}
	cfg.setDefaults()

	assert.Equal(t, 5, cfg.Threshold)
}

func TestNewTieredTTS_OK(t *testing.T) {
	t.Parallel()

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local: &LocalTTS{},
		Cloud: &mockTTSProvider{},
	})
	require.NoError(t, err)
	assert.NotNil(t, tt)
}

// --- 路由测试 ---

func TestTieredTTS_Synthesize_ShortText_UsesLocal(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{}
	// LocalTTS 的 engine 为 nil，Synthesize 会返回"已关闭"错误。
	// 这证明它尝试了本地 TTS 而不是云端。
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 10,
	})
	require.NoError(t, err)

	_, err = tt.Synthesize(context.Background(), "你好", aiface.TTSConfig{})
	// 应尝试本地 → 返回"已关闭"。
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
	// 云端不应被调用。
	assert.Equal(t, 0, cloud.synthesizeCount())
}

func TestTieredTTS_Synthesize_LongText_UsesCloud(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{
		synthesizeResult: []byte{0x01, 0x02},
	}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 5,
	})
	require.NoError(t, err)

	pcm, err := tt.Synthesize(context.Background(), "这是一段很长的文本", aiface.TTSConfig{})
	assert.NoError(t, err)
	assert.Equal(t, []byte{0x01, 0x02}, pcm)
	assert.Equal(t, 1, cloud.synthesizeCount())
}

func TestTieredTTS_Synthesize_ExactThreshold_UsesLocal(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 5,
	})
	require.NoError(t, err)

	// 恰好 5 个字符 → 使用本地。
	_, err = tt.Synthesize(context.Background(), "你好世界啊", aiface.TTSConfig{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "已关闭")
	assert.Equal(t, 0, cloud.synthesizeCount())
}

// --- SynthesizeStream 路由测试 ---

func TestTieredTTS_SynthesizeStream_MixedText(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{
		synthesizeResult: []byte{0xAA, 0xBB},
	}
	// local 为 nil engine，短文本调用会失败并触发 stream 关闭。
	// 为了测试路由逻辑，我们只发送长文本（走云端）。
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 3,
	})
	require.NoError(t, err)

	textCh := make(chan string, 4)
	textCh <- "这是一段长文本用于测试" // > 3 字符 → 云端
	close(textCh)

	audioCh, err := tt.SynthesizeStream(context.Background(), textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	var chunks [][]byte
	for chunk := range audioCh {
		chunks = append(chunks, chunk)
	}

	assert.Len(t, chunks, 1)
	assert.Equal(t, []byte{0xAA, 0xBB}, chunks[0])
}

func TestTieredTTS_SynthesizeStream_ContextCancel(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{
		synthesizeResult: []byte{0xAA, 0xBB},
	}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 3,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	textCh := make(chan string, 4)

	audioCh, err := tt.SynthesizeStream(ctx, textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	cancel()

	// audioCh 应关闭。
	for range audioCh {
		// drain
	}
}

func TestTieredTTS_SynthesizeStream_EmptyText(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{
		synthesizeResult: []byte{0xAA},
	}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 3,
	})
	require.NoError(t, err)

	textCh := make(chan string, 4)
	textCh <- ""            // 空文本应跳过
	textCh <- "这是一段长文本用于测试" // > 3 → 云端
	close(textCh)

	audioCh, err := tt.SynthesizeStream(context.Background(), textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	chunks := drainAudioChannel(audioCh, time.Second)
	assert.Len(t, chunks, 1)
}

func TestTieredTTS_SynthesizeStream_SynthesizeError(t *testing.T) {
	t.Parallel()

	// 短文本走 local，local engine 为 nil 会返回"已关闭"错误 → goroutine 退出。
	cloud := &mockTTSProvider{}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 10,
	})
	require.NoError(t, err)

	textCh := make(chan string, 4)
	textCh <- "你好" // <= 10 → local → 已关闭
	textCh <- "世界" // 不应处理
	close(textCh)

	audioCh, err := tt.SynthesizeStream(context.Background(), textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	// audioCh 应在错误后关闭。
	chunks := drainAudioChannel(audioCh, time.Second)
	assert.Empty(t, chunks)
}

func TestTieredTTS_SynthesizeStream_NilPCMContinues(t *testing.T) {
	t.Parallel()

	// 云端返回 nil pcm → 应 continue 不 panic
	cloud := &mockTTSProvider{
		synthesizeResult: nil, // nil pcm
	}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 3,
	})
	require.NoError(t, err)

	textCh := make(chan string, 4)
	textCh <- "这是一段长文本" // > 3 → cloud → nil result
	close(textCh)

	audioCh, err := tt.SynthesizeStream(context.Background(), textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	chunks := drainAudioChannel(audioCh, time.Second)
	assert.Empty(t, chunks)
}

func TestTieredTTS_SynthesizeStream_CtxCancelOnSend(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{
		synthesizeResult: []byte{0xAA, 0xBB},
	}
	local := &LocalTTS{engine: nil}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local:     local,
		Cloud:     cloud,
		Threshold: 3,
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	textCh := make(chan string, 4)
	// 不读 audioCh，让它阻塞在发送，然后取消 ctx
	audioCh, err := tt.SynthesizeStream(ctx, textCh, aiface.TTSConfig{})
	require.NoError(t, err)

	// 填满 audioCh buffer (cap=8)
	for range 9 {
		textCh <- "这是一段长文本用于测试"
	}
	// 此时第9个应阻塞在发送
	time.Sleep(50 * time.Millisecond)
	cancel()

	// drain
	for range audioCh {
	}
	close(textCh)
}

// --- Cancel ---

func TestTieredTTS_Cancel(t *testing.T) {
	t.Parallel()

	cloud := &mockTTSProvider{cancelErr: errors.New("cancel error")}
	local := &LocalTTS{}

	tt, err := NewTieredTTS(TieredTTSConfig{
		Local: local,
		Cloud: cloud,
	})
	require.NoError(t, err)

	err = tt.Cancel()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cancel error")
}

// --- Mock 实现 ---

type mockTTSProvider struct {
	synthesizeResult []byte
	synthesizeErr    error
	cancelErr        error

	mu         sync.Mutex
	synthCalls int
}

func (m *mockTTSProvider) SynthesizeStream(_ context.Context, textCh <-chan string, _ aiface.TTSConfig) (<-chan []byte, error) {
	audioCh := make(chan []byte, 8)
	go func() {
		defer close(audioCh)
		for text := range textCh {
			if text == "" {
				continue
			}
			m.mu.Lock()
			m.synthCalls++
			result := m.synthesizeResult
			m.mu.Unlock()
			if result != nil {
				audioCh <- result
			}
		}
	}()
	return audioCh, nil
}

func (m *mockTTSProvider) Synthesize(_ context.Context, _ string, _ aiface.TTSConfig) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.synthCalls++
	return m.synthesizeResult, m.synthesizeErr
}

func (m *mockTTSProvider) Cancel() error {
	return m.cancelErr
}

func (m *mockTTSProvider) synthesizeCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.synthCalls
}

// drainAudioChannel 在超时内收集音频通道的所有数据。
func drainAudioChannel(ch <-chan []byte, timeout time.Duration) [][]byte {
	var chunks [][]byte
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case chunk, ok := <-ch:
			if !ok {
				return chunks
			}
			chunks = append(chunks, chunk)
		case <-timer.C:
			return chunks
		}
	}
}
