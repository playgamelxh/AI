package llm

import (
	"ai-agent-skill/internal/models"
	"context"
	"fmt"
	"io"

	"github.com/sashabaranov/go-openai"
)

// OpenAIClient OpenAI兼容客户端
// 支持OpenAI、DeepSeek、Qwen等兼容OpenAI API的提供商
type OpenAIClient struct {
	client   *openai.Client
	config   *models.LLMConfig
	provider models.LLMProvider
}

// NewOpenAIClient 创建OpenAI兼容客户端
// provider: 提供商类型
// config: 模型配置
func NewOpenAIClient(provider models.LLMProvider, config *models.LLMConfig) (*OpenAIClient, error) {
	cfg := openai.DefaultConfig(config.APIKey)
	
	// 如果提供了BaseURL，则使用自定义地址
	if config.BaseURL != "" {
		cfg.BaseURL = config.BaseURL
	}

	client := openai.NewClientWithConfig(cfg)

	return &OpenAIClient{
		client:   client,
		config:   config,
		provider: provider,
	}, nil
}

// Chat 发送聊天请求
func (c *OpenAIClient) Chat(ctx context.Context, messages []models.ChatMessage, options ...ChatOption) (string, error) {
	opts := NewChatOptions(options...)
	
	// 转换消息格式
	oaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		oaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// 构建请求
	model := c.config.Model
	if opts.Model != "" {
		model = opts.Model
	}

	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    oaiMessages,
		Temperature: float32(opts.Temperature),
		MaxTokens:   opts.MaxTokens,
	}

	// 发送请求
	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("chat completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices")
	}

	return resp.Choices[0].Message.Content, nil
}

// ChatStream 流式聊天
func (c *OpenAIClient) ChatStream(ctx context.Context, messages []models.ChatMessage, options ...ChatOption) (<-chan string, error) {
	opts := NewChatOptions(options...)
	
	// 转换消息格式
	oaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		oaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// 构建请求
	model := c.config.Model
	if opts.Model != "" {
		model = opts.Model
	}

	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    oaiMessages,
		Temperature: float32(opts.Temperature),
		MaxTokens:   opts.MaxTokens,
		Stream:      true,
	}

	// 创建流式请求
	stream, err := c.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("create stream failed: %w", err)
	}

	// 创建输出通道
	outChan := make(chan string)

	go func() {
		defer close(outChan)
		defer stream.Close()

		for {
			select {
			case <-ctx.Done():
				return
			default:
				resp, err := stream.Recv()
				if err != nil {
					if err == io.EOF {
						return
					}
					return
				}

				if len(resp.Choices) > 0 {
					content := resp.Choices[0].Delta.Content
					if content != "" {
						outChan <- content
					}
				}
			}
		}
	}()

	return outChan, nil
}

// GetProvider 获取提供商类型
func (c *OpenAIClient) GetProvider() models.LLMProvider {
	return c.provider
}

// Validate 验证配置
func (c *OpenAIClient) Validate() error {
	if c.config.APIKey == "" && c.provider != models.LLMProviderOllama {
		return fmt.Errorf("API key is required for provider %s", c.provider)
	}
	if c.config.Model == "" {
		return fmt.Errorf("model is required")
	}
	return nil
}
