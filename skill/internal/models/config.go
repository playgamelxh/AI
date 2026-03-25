package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// LLMProvider 定义支持的大模型提供商类型
type LLMProvider string

const (
	// LLMProviderOpenAI OpenAI/ChatGPT 提供商
	LLMProviderOpenAI LLMProvider = "openai"
	// LLMProviderOllama Ollama 本地模型提供商
	LLMProviderOllama LLMProvider = "ollama"
	// LLMProviderDeepSeek DeepSeek 提供商
	LLMProviderDeepSeek LLMProvider = "deepseek"
	// LLMProviderDoubao 豆包提供商
	LLMProviderDoubao LLMProvider = "doubao"
	// LLMProviderQwen 通义千问提供商
	LLMProviderQwen LLMProvider = "qwen"
	// LLMProviderGemini Google DeepMind（Gemini）提供商
	LLMProviderGemini LLMProvider = "gemini"
	// LLMProviderClaude Anthropic（Claude）提供商
	LLMProviderClaude LLMProvider = "claude"
	// LLMProviderLLaMA Meta（LLaMA）提供商
	LLMProviderLLaMA LLMProvider = "llama"
	// LLMProviderGrok XAI（Grok）提供商
	LLMProviderGrok LLMProvider = "grok"
	// LLMProviderYuanbao 腾讯（元宝）提供商
	LLMProviderYuanbao LLMProvider = "yuanbao"
	// LLMProviderGLM 智谱 AI（GLM）提供商
	LLMProviderGLM LLMProvider = "glm"
	// LLMProviderKimi 月之暗面（Kimi）提供商
	LLMProviderKimi LLMProvider = "kimi"
)

// LLMConfig 大模型配置
type LLMConfig struct {
	// ID 配置唯一标识
	ID string `json:"id" yaml:"id" gorm:"primaryKey;type:varchar(255)"`
	// Provider 模型提供商类型
	Provider LLMProvider `json:"provider" yaml:"provider" gorm:"type:varchar(50);index"`
	// Name 配置名称
	Name string `json:"name" yaml:"name" gorm:"type:varchar(255)"`
	// APIKey API密钥
	APIKey string `json:"api_key,omitempty" yaml:"api_key,omitempty" gorm:"type:text"`
	// BaseURL API基础地址
	BaseURL string `json:"base_url,omitempty" yaml:"base_url,omitempty" gorm:"type:varchar(500)"`
	// Model 模型名称
	Model string `json:"model" yaml:"model" gorm:"type:varchar(255)"`
	// Temperature 温度参数
	Temperature float64 `json:"temperature,omitempty" yaml:"temperature,omitempty" gorm:"default:0.7"`
	// MaxTokens 最大token数
	MaxTokens int `json:"max_tokens,omitempty" yaml:"max_tokens,omitempty" gorm:"default:2048"`
	// Enabled 是否启用
	Enabled bool `json:"enabled" yaml:"enabled" gorm:"default:true"`
	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at" yaml:"-"`
	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at" yaml:"-"`
	// DeletedAt 删除时间（软删除）
	DeletedAt gorm.DeletedAt `json:"-" yaml:"-" gorm:"index"`
}

// NewLLMConfig 创建新的大模型配置
func NewLLMConfig(provider LLMProvider, name, apiKey, baseURL, model string) *LLMConfig {
	return &LLMConfig{
		ID:          uuid.New().String(),
		Provider:    provider,
		Name:        name,
		APIKey:      apiKey,
		BaseURL:     baseURL,
		Model:       model,
		Temperature: 0.7,
		MaxTokens:   2048,
		Enabled:     true,
	}
}

// SkillType 技能类型
type SkillType string

const (
	// SkillTypeCommand 系统命令执行技能
	SkillTypeCommand SkillType = "command"
	// SkillTypeFile 文件操作技能
	SkillTypeFile SkillType = "file"
	// SkillTypeApplication 应用程序操作技能
	SkillTypeApplication SkillType = "application"
)

// Skill 技能定义
type Skill struct {
	// ID 技能唯一标识
	ID string `json:"id" yaml:"id" gorm:"primaryKey;type:varchar(255)"`
	// Name 技能名称
	Name string `json:"name" yaml:"name" gorm:"type:varchar(255)"`
	// Type 技能类型
	Type SkillType `json:"type" yaml:"type" gorm:"type:varchar(50);index"`
	// Description 技能描述
	Description string `json:"description" yaml:"description" gorm:"type:text"`
	// Enabled 是否启用
	Enabled bool `json:"enabled" yaml:"enabled" gorm:"default:true"`
	// Config 技能配置（JSON格式）
	ConfigJSON string `json:"-" yaml:"-" gorm:"type:text;column:config_json"`
	// Config 技能配置（内存中使用）
	Config map[string]interface{} `json:"config,omitempty" yaml:"config,omitempty" gorm:"-"`
	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at" yaml:"-"`
	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at" yaml:"-"`
	// DeletedAt 删除时间（软删除）
	DeletedAt gorm.DeletedAt `json:"-" yaml:"-" gorm:"index"`
}

// NewSkill 创建新技能
func NewSkill(name string, skillType SkillType, description string) *Skill {
	return &Skill{
		ID:          uuid.New().String(),
		Name:        name,
		Type:        skillType,
		Description: description,
		Enabled:     true,
		Config:      make(map[string]interface{}),
	}
}

// ChatMessage 聊天消息
type ChatMessage struct {
	// Role 角色 (user, assistant, system)
	Role string `json:"role"`
	// Content 消息内容
	Content string `json:"content"`
}

// ChatRequest 聊天请求
type ChatRequest struct {
	// ConfigID 使用的模型配置ID
	ConfigID string `json:"config_id"`
	// Messages 消息列表
	Messages []ChatMessage `json:"messages"`
	// Stream 是否流式输出
	Stream bool `json:"stream,omitempty"`
}

// ChatResponse 聊天响应
type ChatResponse struct {
	// Success 是否成功
	Success bool `json:"success"`
	// Message 响应消息
	Message string `json:"message,omitempty"`
	// Error 错误信息
	Error string `json:"error,omitempty"`
}

// APIResponse 统一API响应
type APIResponse struct {
	// Success 是否成功
	Success bool `json:"success"`
	// Data 响应数据
	Data interface{} `json:"data,omitempty"`
	// Error 错误信息
	Error string `json:"error,omitempty"`
	// Message 提示信息
	Message string `json:"message,omitempty"`
}

// NewSuccessResponse 创建成功响应
func NewSuccessResponse(data interface{}) APIResponse {
	return APIResponse{
		Success: true,
		Data:    data,
	}
}

// NewErrorResponse 创建错误响应
func NewErrorResponse(message string) APIResponse {
	return APIResponse{
		Success: false,
		Error:   message,
	}
}

// NewMessageResponse 创建带消息的响应
func NewMessageResponse(message string, data interface{}) APIResponse {
	return APIResponse{
		Success: true,
		Data:    data,
		Message: message,
	}
}

// Agent AI智能体
type Agent struct {
	// ID 智能体唯一标识
	ID string `json:"id" gorm:"primaryKey;type:varchar(255)"`
	// Name 智能体名称
	Name string `json:"name" gorm:"type:varchar(255)"`
	// Description 智能体描述
	Description string `json:"description" gorm:"type:text"`
	// LLMConfigID 绑定的大模型配置ID
	LLMConfigID string `json:"llm_config_id" gorm:"type:varchar(255);index"`
	// LLMConfig 绑定的大模型配置
	LLMConfig *LLMConfig `json:"llm_config,omitempty" gorm:"foreignKey:LLMConfigID"`
	// SkillIDsJSON 绑定的技能ID列表（JSON格式）
	SkillIDsJSON string `json:"-" gorm:"type:text;column:skill_ids_json"`
	// SkillIDs 绑定的技能ID列表（内存中使用）
	SkillIDs []string `json:"skill_ids" gorm:"-"`
	// Enabled 是否启用
	Enabled bool `json:"enabled" gorm:"default:true"`
	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at"`
	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at"`
	// DeletedAt 删除时间（软删除）
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`
}

// NewAgent 创建新智能体
func NewAgent(name, description, llmConfigID string) *Agent {
	return &Agent{
		ID:          uuid.New().String(),
		Name:        name,
		Description: description,
		LLMConfigID: llmConfigID,
		SkillIDs:    []string{},
		Enabled:     true,
	}
}

// TaskType 任务类型
type TaskType string

const (
	// TaskTypeOnce 一次性任务
	TaskTypeOnce TaskType = "once"
	// TaskTypePeriodic 周期性任务
	TaskTypePeriodic TaskType = "periodic"
)

// TaskStatus 任务状态
type TaskStatus string

const (
	// TaskStatusPending 待执行
	TaskStatusPending TaskStatus = "pending"
	// TaskStatusRunning 执行中
	TaskStatusRunning TaskStatus = "running"
	// TaskStatusCompleted 已完成
	TaskStatusCompleted TaskStatus = "completed"
	// TaskStatusFailed 失败
	TaskStatusFailed TaskStatus = "failed"
	// TaskStatusPaused 已暂停
	TaskStatusPaused TaskStatus = "paused"
)

// Task 任务
type Task struct {
	// ID 任务唯一标识
	ID string `json:"id" gorm:"primaryKey;type:varchar(255)"`
	// AgentID 关联的智能体ID
	AgentID string `json:"agent_id" gorm:"type:varchar(255);index"`
	// Agent 关联的智能体
	Agent *Agent `json:"agent,omitempty" gorm:"foreignKey:AgentID"`
	// Name 任务名称
	Name string `json:"name" gorm:"type:varchar(255)"`
	// Description 任务描述
	Description string `json:"description" gorm:"type:text"`
	// Type 任务类型
	Type TaskType `json:"type" gorm:"type:varchar(50);index"`
	// Status 任务状态
	Status TaskStatus `json:"status" gorm:"type:varchar(50);index"`
	// CronExpr Cron表达式（周期性任务使用）
	CronExpr string `json:"cron_expr,omitempty" gorm:"type:varchar(100)"`
	// ScheduledTime 计划执行时间
	ScheduledTime *time.Time `json:"scheduled_time,omitempty" gorm:"index"`
	// LastRunTime 最后一次执行时间
	LastRunTime *time.Time `json:"last_run_time,omitempty"`
	// NextRunTime 下一次执行时间
	NextRunTime *time.Time `json:"next_run_time,omitempty" gorm:"index"`
	// ContentJSON 任务内容（JSON格式）
	ContentJSON string `json:"-" gorm:"type:text;column:content_json"`
	// Content 任务内容（内存中使用）
	Content map[string]interface{} `json:"content" gorm:"-"`
	// ResultJSON 任务结果（JSON格式）
	ResultJSON string `json:"-" gorm:"type:text;column:result_json"`
	// Result 任务结果（内存中使用）
	Result map[string]interface{} `json:"result,omitempty" gorm:"-"`
	// ErrorMessage 错误信息
	ErrorMessage string `json:"error_message,omitempty" gorm:"type:text"`
	// RetryCount 重试次数
	RetryCount int `json:"retry_count" gorm:"default:0"`
	// MaxRetries 最大重试次数
	MaxRetries int `json:"max_retries" gorm:"default:3"`
	// Enabled 是否启用
	Enabled bool `json:"enabled" gorm:"default:true"`
	// CreatedAt 创建时间
	CreatedAt time.Time `json:"created_at"`
	// UpdatedAt 更新时间
	UpdatedAt time.Time `json:"updated_at"`
	// DeletedAt 删除时间（软删除）
	DeletedAt gorm.DeletedAt `json:"-" gorm:"index"`
}

// NewTask 创建新任务
func NewTask(agentID, name, description string, taskType TaskType) *Task {
	now := time.Now()
	return &Task{
		ID:          uuid.New().String(),
		AgentID:     agentID,
		Name:        name,
		Description: description,
		Type:        taskType,
		Status:      TaskStatusPending,
		Content:     make(map[string]interface{}),
		Result:      make(map[string]interface{}),
		MaxRetries:  3,
		Enabled:     true,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
}
