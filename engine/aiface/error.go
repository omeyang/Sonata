package aiface

import (
	"errors"
	"fmt"
	"time"
)

// ErrorKind 区分提供者错误的类型。
type ErrorKind int

const (
	// ErrKindTemporary 表示可重试的临时错误（网络抖动、限流）。
	ErrKindTemporary ErrorKind = iota
	// ErrKindPermanent 表示不可重试的永久错误（认证失败、参数错误）。
	ErrKindPermanent
	// ErrKindTimeout 表示超时错误。
	ErrKindTimeout
)

// Error 是 AI 提供者返回的结构化错误。
type Error struct {
	Kind      ErrorKind
	Provider  string // 提供者名称（如 "qwen_asr", "deepseek", "dashscope_tts"）。
	Operation string // 操作名称（如 "connect", "feed_audio", "synthesize"）。
	Cause     error  // 底层错误。
	Timestamp time.Time
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s.%s: %v", e.Provider, e.Operation, e.Cause)
}

func (e *Error) Unwrap() error {
	return e.Cause
}

// IsTemporary 检查错误是否为可重试的临时错误。
func (e *Error) IsTemporary() bool {
	return e.Kind == ErrKindTemporary
}

// IsTimeout 检查错误是否为超时错误。
func (e *Error) IsTimeout() bool {
	return e.Kind == ErrKindTimeout
}

// NewTemporaryError 创建临时错误。
func NewTemporaryError(provider, operation string, cause error) *Error {
	return &Error{
		Kind:      ErrKindTemporary,
		Provider:  provider,
		Operation: operation,
		Cause:     cause,
		Timestamp: time.Now(),
	}
}

// NewPermanentError 创建永久错误。
func NewPermanentError(provider, operation string, cause error) *Error {
	return &Error{
		Kind:      ErrKindPermanent,
		Provider:  provider,
		Operation: operation,
		Cause:     cause,
		Timestamp: time.Now(),
	}
}

// NewTimeoutError 创建超时错误。
func NewTimeoutError(provider, operation string, cause error) *Error {
	return &Error{
		Kind:      ErrKindTimeout,
		Provider:  provider,
		Operation: operation,
		Cause:     cause,
		Timestamp: time.Now(),
	}
}

// IsError 检查错误链中是否包含 Error。
func IsError(err error) bool {
	var pe *Error
	return errors.As(err, &pe)
}

// AsError 从错误链中提取 Error。
func AsError(err error) (*Error, bool) {
	var pe *Error
	if errors.As(err, &pe) {
		return pe, true
	}
	return nil, false
}

// Sentinel errors for common provider failures.
var (
	// ErrProviderUnavailable 表示提供者服务不可用。
	ErrProviderUnavailable = errors.New("provider: service unavailable")
	// ErrProviderAuth 表示认证失败。
	ErrProviderAuth = errors.New("provider: authentication failed")
	// ErrProviderRateLimit 表示请求被限流。
	ErrProviderRateLimit = errors.New("provider: rate limited")
)
