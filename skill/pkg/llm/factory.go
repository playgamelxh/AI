package llm

import (
	"ai-agent-skill/internal/models"
	"fmt"
)

// ClientFactory LLM客户端工厂
// 负责根据配置创建对应的LLM客户端
type ClientFactory struct{}

// NewClientFactory 创建客户端工厂
func NewClientFactory() *ClientFactory {
	return &ClientFactory{}
}

// CreateClient 根据配置创建LLM客户端
// config: 大模型配置
// 返回: LLM客户端实例，错误信息
func (f *ClientFactory) CreateClient(config *models.LLMConfig) (LLMClient, error) {
	if config == nil {
		return nil, fmt.Errorf("config is nil")
	}

	switch config.Provider {
	case models.LLMProviderOpenAI:
		return f.createOpenAIClient(config)
	case models.LLMProviderOllama:
		return f.createOllamaClient(config)
	case models.LLMProviderDeepSeek:
		return f.createDeepSeekClient(config)
	case models.LLMProviderDoubao:
		return f.createDoubaoClient(config)
	case models.LLMProviderQwen:
		return f.createQwenClient(config)
	case models.LLMProviderGemini:
		return f.createGeminiClient(config)
	case models.LLMProviderClaude:
		return f.createClaudeClient(config)
	case models.LLMProviderLLaMA:
		return f.createLLaMAClient(config)
	case models.LLMProviderGrok:
		return f.createGrokClient(config)
	case models.LLMProviderYuanbao:
		return f.createYuanbaoClient(config)
	case models.LLMProviderGLM:
		return f.createGLMClient(config)
	case models.LLMProviderKimi:
		return f.createKimiClient(config)
	default:
		return nil, fmt.Errorf("unsupported provider: %s", config.Provider)
	}
}

// createOpenAIClient 创建OpenAI客户端
func (f *ClientFactory) createOpenAIClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.openai.com/v1"
	}
	if config.Model == "" {
		config.Model = "gpt-3.5-turbo"
	}
	return NewOpenAIClient(models.LLMProviderOpenAI, config)
}

// createOllamaClient 创建Ollama客户端
func (f *ClientFactory) createOllamaClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "http://localhost:11434/v1"
	}
	if config.Model == "" {
		config.Model = "llama2"
	}
	return NewOpenAIClient(models.LLMProviderOllama, config)
}

// createDeepSeekClient 创建DeepSeek客户端
func (f *ClientFactory) createDeepSeekClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.deepseek.com/v1"
	}
	if config.Model == "" {
		config.Model = "deepseek-chat"
	}
	return NewOpenAIClient(models.LLMProviderDeepSeek, config)
}

// createDoubaoClient 创建豆包客户端
func (f *ClientFactory) createDoubaoClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://ark.cn-beijing.volces.com/api/v3"
	}
	if config.Model == "" {
		config.Model = "ep-20241203103756-9k8x2"
	}
	return NewOpenAIClient(models.LLMProviderDoubao, config)
}

// createQwenClient 创建通义千问客户端
func (f *ClientFactory) createQwenClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	if config.Model == "" {
		config.Model = "qwen-turbo"
	}
	return NewOpenAIClient(models.LLMProviderQwen, config)
}

// createGeminiClient 创建Gemini客户端
func (f *ClientFactory) createGeminiClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	if config.Model == "" {
		config.Model = "gemini-pro"
	}
	return NewOpenAIClient(models.LLMProviderGemini, config)
}

// createClaudeClient 创建Claude客户端
func (f *ClientFactory) createClaudeClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.anthropic.com/v1"
	}
	if config.Model == "" {
		config.Model = "claude-3-opus-20240229"
	}
	return NewOpenAIClient(models.LLMProviderClaude, config)
}

// createLLaMAClient 创建LLaMA客户端
func (f *ClientFactory) createLLaMAClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "http://localhost:11434/v1"
	}
	if config.Model == "" {
		config.Model = "llama3"
	}
	return NewOpenAIClient(models.LLMProviderLLaMA, config)
}

// createGrokClient 创建Grok客户端
func (f *ClientFactory) createGrokClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.x.ai/v1"
	}
	if config.Model == "" {
		config.Model = "grok-2"
	}
	return NewOpenAIClient(models.LLMProviderGrok, config)
}

// createYuanbaoClient 创建元宝客户端
func (f *ClientFactory) createYuanbaoClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.yuanbao.qq.com/v1"
	}
	if config.Model == "" {
		config.Model = "yuanbao-pro"
	}
	return NewOpenAIClient(models.LLMProviderYuanbao, config)
}

// createGLMClient 创建GLM客户端
func (f *ClientFactory) createGLMClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://open.bigmodel.cn/api/paas/v4"
	}
	if config.Model == "" {
		config.Model = "glm-4"
	}
	return NewOpenAIClient(models.LLMProviderGLM, config)
}

// createKimiClient 创建Kimi客户端
func (f *ClientFactory) createKimiClient(config *models.LLMConfig) (LLMClient, error) {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.moonshot.cn/v1"
	}
	if config.Model == "" {
		config.Model = "moonshot-v1-8k"
	}
	return NewOpenAIClient(models.LLMProviderKimi, config)
}

// GetDefaultProviders 获取支持的提供商列表
func GetDefaultProviders() []models.LLMProvider {
	return []models.LLMProvider{
		models.LLMProviderOpenAI,
		models.LLMProviderOllama,
		models.LLMProviderDeepSeek,
		models.LLMProviderDoubao,
		models.LLMProviderQwen,
		models.LLMProviderGemini,
		models.LLMProviderClaude,
		models.LLMProviderLLaMA,
		models.LLMProviderGrok,
		models.LLMProviderYuanbao,
		models.LLMProviderGLM,
		models.LLMProviderKimi,
	}
}

// GetProviderInfo 获取提供商信息
func GetProviderInfo(provider models.LLMProvider) map[string]string {
	info := map[string]map[string]string{
		string(models.LLMProviderOpenAI): {
			"name":          "OpenAI / ChatGPT",
			"default_url":   "https://api.openai.com/v1",
			"default_model": "gpt-3.5-turbo",
		},
		string(models.LLMProviderOllama): {
			"name":          "Ollama (本地)",
			"default_url":   "http://localhost:11434/v1",
			"default_model": "llama2",
		},
		string(models.LLMProviderDeepSeek): {
			"name":          "DeepSeek",
			"default_url":   "https://api.deepseek.com/v1",
			"default_model": "deepseek-chat",
		},
		string(models.LLMProviderDoubao): {
			"name":          "豆包 (火山引擎)",
			"default_url":   "https://ark.cn-beijing.volces.com/api/v3",
			"default_model": "ep-20241203103756-9k8x2",
		},
		string(models.LLMProviderQwen): {
			"name":          "通义千问",
			"default_url":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
			"default_model": "qwen-turbo",
		},
		string(models.LLMProviderGemini): {
			"name":          "Google DeepMind (Gemini)",
			"default_url":   "https://generativelanguage.googleapis.com/v1beta",
			"default_model": "gemini-pro",
		},
		string(models.LLMProviderClaude): {
			"name":          "Anthropic (Claude)",
			"default_url":   "https://api.anthropic.com/v1",
			"default_model": "claude-3-opus-20240229",
		},
		string(models.LLMProviderLLaMA): {
			"name":          "Meta (LLaMA)",
			"default_url":   "http://localhost:11434/v1",
			"default_model": "llama3",
		},
		string(models.LLMProviderGrok): {
			"name":          "XAI (Grok)",
			"default_url":   "https://api.x.ai/v1",
			"default_model": "grok-2",
		},
		string(models.LLMProviderYuanbao): {
			"name":          "腾讯 (元宝)",
			"default_url":   "https://api.yuanbao.qq.com/v1",
			"default_model": "yuanbao-pro",
		},
		string(models.LLMProviderGLM): {
			"name":          "智谱 AI (GLM)",
			"default_url":   "https://open.bigmodel.cn/api/paas/v4",
			"default_model": "glm-4",
		},
		string(models.LLMProviderKimi): {
			"name":          "月之暗面 (Kimi)",
			"default_url":   "https://api.moonshot.cn/v1",
			"default_model": "moonshot-v1-8k",
		},
	}

	if p, ok := info[string(provider)]; ok {
		return p
	}
	return nil
}
