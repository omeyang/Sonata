package aiface

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestErrorKindConstants 验证 ErrorKind 常量值。
func TestErrorKindConstants(t *testing.T) {
	t.Parallel()

	assert.Equal(t, ErrorKind(0), ErrKindTemporary)
	assert.Equal(t, ErrorKind(1), ErrKindPermanent)
	assert.Equal(t, ErrorKind(2), ErrKindTimeout)
}

// TestErrorError 验证 Error.Error() 的格式化输出。
func TestErrorError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		err      *Error
		expected string
	}{
		{
			name: "临时错误格式",
			err: &Error{
				Kind:      ErrKindTemporary,
				Provider:  "qwen_asr",
				Operation: "connect",
				Cause:     errors.New("connection refused"),
			},
			expected: "qwen_asr.connect: connection refused",
		},
		{
			name: "永久错误格式",
			err: &Error{
				Kind:      ErrKindPermanent,
				Provider:  "deepseek",
				Operation: "generate",
				Cause:     errors.New("auth failed"),
			},
			expected: "deepseek.generate: auth failed",
		},
		{
			name: "超时错误格式",
			err: &Error{
				Kind:      ErrKindTimeout,
				Provider:  "dashscope_tts",
				Operation: "synthesize",
				Cause:     errors.New("deadline exceeded"),
			},
			expected: "dashscope_tts.synthesize: deadline exceeded",
		},
		{
			name: "空提供者和操作",
			err: &Error{
				Kind:      ErrKindTemporary,
				Provider:  "",
				Operation: "",
				Cause:     errors.New("unknown"),
			},
			expected: ".: unknown",
		},
		{
			name: "nil cause",
			err: &Error{
				Kind:      ErrKindTemporary,
				Provider:  "test",
				Operation: "op",
				Cause:     nil,
			},
			expected: "test.op: <nil>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, tt.err.Error())
		})
	}
}

// TestErrorUnwrap 验证 Error.Unwrap() 返回底层错误。
func TestErrorUnwrap(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		cause error
	}{
		{"有底层错误", errors.New("root cause")},
		{"nil底层错误", nil},
		{"哨兵错误", ErrProviderUnavailable},
		{"嵌套错误", fmt.Errorf("wrap: %w", errors.New("inner"))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := &Error{Cause: tt.cause}
			assert.Equal(t, tt.cause, e.Unwrap())
		})
	}
}

// TestErrorIsTemporary 验证 IsTemporary() 对各种 ErrorKind 的判断。
func TestErrorIsTemporary(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		kind ErrorKind
		want bool
	}{
		{"临时错误返回true", ErrKindTemporary, true},
		{"永久错误返回false", ErrKindPermanent, false},
		{"超时错误返回false", ErrKindTimeout, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := &Error{Kind: tt.kind}
			assert.Equal(t, tt.want, e.IsTemporary())
		})
	}
}

// TestErrorIsTimeout 验证 IsTimeout() 对各种 ErrorKind 的判断。
func TestErrorIsTimeout(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		kind ErrorKind
		want bool
	}{
		{"超时错误返回true", ErrKindTimeout, true},
		{"临时错误返回false", ErrKindTemporary, false},
		{"永久错误返回false", ErrKindPermanent, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			e := &Error{Kind: tt.kind}
			assert.Equal(t, tt.want, e.IsTimeout())
		})
	}
}

// TestNewTemporaryError 验证 NewTemporaryError 构造函数。
func TestNewTemporaryError(t *testing.T) {
	t.Parallel()

	cause := errors.New("network timeout")
	e := NewTemporaryError("qwen_asr", "connect", cause)

	require.NotNil(t, e)
	assert.Equal(t, ErrKindTemporary, e.Kind)
	assert.Equal(t, "qwen_asr", e.Provider)
	assert.Equal(t, "connect", e.Operation)
	assert.Equal(t, cause, e.Cause)
	assert.False(t, e.Timestamp.IsZero(), "时间戳应自动设置")
	assert.True(t, e.IsTemporary())
	assert.False(t, e.IsTimeout())
}

// TestNewPermanentError 验证 NewPermanentError 构造函数。
func TestNewPermanentError(t *testing.T) {
	t.Parallel()

	cause := errors.New("invalid api key")
	e := NewPermanentError("deepseek", "generate", cause)

	require.NotNil(t, e)
	assert.Equal(t, ErrKindPermanent, e.Kind)
	assert.Equal(t, "deepseek", e.Provider)
	assert.Equal(t, "generate", e.Operation)
	assert.Equal(t, cause, e.Cause)
	assert.False(t, e.Timestamp.IsZero())
	assert.False(t, e.IsTemporary())
	assert.False(t, e.IsTimeout())
}

// TestNewTimeoutError 验证 NewTimeoutError 构造函数。
func TestNewTimeoutError(t *testing.T) {
	t.Parallel()

	cause := errors.New("context deadline exceeded")
	e := NewTimeoutError("dashscope_tts", "synthesize", cause)

	require.NotNil(t, e)
	assert.Equal(t, ErrKindTimeout, e.Kind)
	assert.Equal(t, "dashscope_tts", e.Provider)
	assert.Equal(t, "synthesize", e.Operation)
	assert.Equal(t, cause, e.Cause)
	assert.False(t, e.Timestamp.IsZero())
	assert.False(t, e.IsTemporary())
	assert.True(t, e.IsTimeout())
}

// TestIsError 验证 IsError 在错误链中检测 *Error。
func TestIsError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "直接的Error指针",
			err:  NewTemporaryError("p", "op", errors.New("x")),
			want: true,
		},
		{
			name: "包装后的Error",
			err:  fmt.Errorf("outer: %w", NewPermanentError("p", "op", errors.New("x"))),
			want: true,
		},
		{
			name: "多层包装的Error",
			err:  fmt.Errorf("l2: %w", fmt.Errorf("l1: %w", NewTimeoutError("p", "op", errors.New("x")))),
			want: true,
		},
		{
			name: "普通错误",
			err:  errors.New("not a provider error"),
			want: false,
		},
		{
			name: "nil错误",
			err:  nil,
			want: false,
		},
		{
			name: "哨兵错误",
			err:  ErrProviderUnavailable,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, IsError(tt.err))
		})
	}
}

// TestAsError 验证 AsError 从错误链中提取 *Error。
func TestAsError(t *testing.T) {
	t.Parallel()

	t.Run("直接提取", func(t *testing.T) {
		t.Parallel()
		original := NewTemporaryError("asr", "feed", errors.New("broken"))
		got, ok := AsError(original)
		require.True(t, ok)
		assert.Equal(t, original, got)
	})

	t.Run("从包装链提取", func(t *testing.T) {
		t.Parallel()
		original := NewPermanentError("tts", "synth", errors.New("fail"))
		wrapped := fmt.Errorf("wrap: %w", original)
		got, ok := AsError(wrapped)
		require.True(t, ok)
		assert.Equal(t, original, got)
	})

	t.Run("非Error类型返回nil和false", func(t *testing.T) {
		t.Parallel()
		got, ok := AsError(errors.New("plain"))
		assert.False(t, ok)
		assert.Nil(t, got)
	})

	t.Run("nil错误返回nil和false", func(t *testing.T) {
		t.Parallel()
		got, ok := AsError(nil)
		assert.False(t, ok)
		assert.Nil(t, got)
	})
}

// TestSentinelErrors 验证哨兵错误的存在和可区分性。
func TestSentinelErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		sentinel error
		msg      string
	}{
		{"服务不可用", ErrProviderUnavailable, "provider: service unavailable"},
		{"认证失败", ErrProviderAuth, "provider: authentication failed"},
		{"限流", ErrProviderRateLimit, "provider: rate limited"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.msg, tt.sentinel.Error())
		})
	}

	// 验证哨兵错误互不相等。
	t.Run("哨兵错误互不相等", func(t *testing.T) {
		t.Parallel()
		assert.NotEqual(t, ErrProviderUnavailable, ErrProviderAuth)
		assert.NotEqual(t, ErrProviderAuth, ErrProviderRateLimit)
		assert.NotEqual(t, ErrProviderUnavailable, ErrProviderRateLimit)
	})
}

// TestErrorUnwrapChain 验证 errors.Is 可以通过 Unwrap 穿透 Error。
func TestErrorUnwrapChain(t *testing.T) {
	t.Parallel()

	e := NewTemporaryError("asr", "connect", ErrProviderUnavailable)
	assert.True(t, errors.Is(e, ErrProviderUnavailable), "应能通过 Unwrap 链匹配哨兵错误")
	assert.False(t, errors.Is(e, ErrProviderAuth), "不应匹配不相关的哨兵错误")

	// 再包装一层。
	wrapped := fmt.Errorf("session: %w", e)
	assert.True(t, errors.Is(wrapped, ErrProviderUnavailable))
}

// TestErrorAsInterface 验证 Error 实现 error 接口。
func TestErrorAsInterface(t *testing.T) {
	t.Parallel()

	var err error = NewTemporaryError("p", "o", errors.New("x"))
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "p.o")
}

// FuzzErrorString 模糊测试 Error.Error() 不会 panic。
func FuzzErrorString(f *testing.F) {
	f.Add("provider", "operation", "cause")
	f.Add("", "", "")
	f.Add("a.b", "c/d", "e\nf")
	f.Add("中文提供者", "中文操作", "中文原因")

	f.Fuzz(func(t *testing.T, provider, operation, cause string) {
		e := &Error{
			Kind:      ErrKindTemporary,
			Provider:  provider,
			Operation: operation,
			Cause:     errors.New(cause),
		}
		got := e.Error()
		if got == "" {
			t.Error("Error() 不应返回空字符串")
		}
	})
}

// FuzzNewTemporaryError 模糊测试 NewTemporaryError 不会 panic。
func FuzzNewTemporaryError(f *testing.F) {
	f.Add("p", "o", "c")
	f.Add("", "", "")

	f.Fuzz(func(t *testing.T, provider, operation, cause string) {
		e := NewTemporaryError(provider, operation, errors.New(cause))
		if e == nil {
			t.Fatal("NewTemporaryError 不应返回 nil")
		}
		if e.Kind != ErrKindTemporary {
			t.Errorf("Kind = %v, want ErrKindTemporary", e.Kind)
		}
		if e.Error() == "" {
			t.Error("Error() 不应返回空字符串")
		}
	})
}
