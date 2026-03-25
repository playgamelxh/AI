package llm

import (
	"ai-agent-skill/internal/models"
	"context"
)

// LLMClient 大模型客户端接口
// 定义统一的大模型调用接口，支持多种提供商实现
type LLMClient interface {
	// Chat 发送聊天请求
	// ctx: 上下文，用于超时控制
	// messages: 聊天消息列表
	// options: 可选参数
	// 返回: 响应消息，错误信息
	Chat(ctx context.Context, messages []models.ChatMessage, options ...ChatOption) (string, error)

	// ChatStream 流式聊天
	// ctx: 上下文
	// messages: 聊天消息列表
	// options: 可选参数
	// 返回: 消息通道，错误
	ChatStream(ctx context.Context, messages []models.ChatMessage, options ...ChatOption) (<-chan string, error)

	// GetProvider 获取提供商类型
	GetProvider() models.LLMProvider

	// Validate 验证配置是否有效
	Validate() error
}

// ChatOption 聊天选项
type ChatOption func(*ChatOptions)

// ChatOptions 聊天选项配置
type ChatOptions struct {
	// Temperature 温度参数
	Temperature float64
	// MaxTokens 最大token数
	MaxTokens int
	// Model 模型名称
	Model string
}

// WithTemperature 设置温度
func WithTemperature(temp float64) ChatOption {
	return func(opts *ChatOptions) {
		opts.Temperature = temp
	}
}

// WithMaxTokens 设置最大token数
func WithMaxTokens(max int) ChatOption {
	return func(opts *ChatOptions) {
		opts.MaxTokens = max
	}
}

// WithModel 设置模型
func WithModel(model string) ChatOption {
	return func(opts *ChatOptions) {
		opts.Model = model
	}
}

// NewChatOptions 创建聊天选项
func NewChatOptions(options ...ChatOption) *ChatOptions {
	opts := &ChatOptions{
		Temperature: 0.7,
		MaxTokens:   2048,
	}
	for _, opt := range options {
		opt(opts)
	}
	return opts
}
