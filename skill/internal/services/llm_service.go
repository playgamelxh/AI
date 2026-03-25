package services

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/pkg/database"
	"ai-agent-skill/pkg/llm"
	"sync"
)

// LLMService 大模型配置管理服务
type LLMService struct {
	clients       map[string]llm.LLMClient
	clientFactory *llm.ClientFactory
	mu            sync.RWMutex
}

// NewLLMService 创建LLM服务
func NewLLMService() *LLMService {
	return &LLMService{
		clients:       make(map[string]llm.LLMClient),
		clientFactory: llm.NewClientFactory(),
	}
}

// ListConfigs 获取所有配置
func (s *LLMService) ListConfigs() ([]*models.LLMConfig, error) {
	db := database.GetDB()
	var configs []*models.LLMConfig
	err := db.Find(&configs).Error
	return configs, err
}

// GetConfig 获取单个配置
func (s *LLMService) GetConfig(id string) (*models.LLMConfig, bool, error) {
	db := database.GetDB()
	var config models.LLMConfig
	err := db.First(&config, "id = ?", id).Error
	if err != nil {
		return nil, false, err
	}
	return &config, true, nil
}

// CreateConfig 创建配置
func (s *LLMService) CreateConfig(cfg *models.LLMConfig) (*models.LLMConfig, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if cfg.ID == "" {
		cfg.ID = cfg.Name
	}

	client, err := s.clientFactory.CreateClient(cfg)
	if err != nil {
		return nil, err
	}

	db := database.GetDB()
	if err := db.Create(cfg).Error; err != nil {
		return nil, err
	}

	s.clients[cfg.ID] = client
	return cfg, nil
}

// UpdateConfig 更新配置
func (s *LLMService) UpdateConfig(id string, cfg *models.LLMConfig) (*models.LLMConfig, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	var existing models.LLMConfig
	if err := db.First(&existing, "id = ?", id).Error; err != nil {
		return nil, nil
	}

	cfg.ID = id
	client, err := s.clientFactory.CreateClient(cfg)
	if err != nil {
		return nil, err
	}

	if err := db.Save(cfg).Error; err != nil {
		return nil, err
	}

	s.clients[id] = client
	return cfg, nil
}

// DeleteConfig 删除配置
func (s *LLMService) DeleteConfig(id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	result := db.Delete(&models.LLMConfig{}, "id = ?", id)
	if result.Error != nil {
		return false, result.Error
	}
	if result.RowsAffected == 0 {
		return false, nil
	}

	delete(s.clients, id)
	return true, nil
}

// ToggleConfig 切换配置启用状态
func (s *LLMService) ToggleConfig(id string) (*models.LLMConfig, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	var config models.LLMConfig
	if err := db.First(&config, "id = ?", id).Error; err != nil {
		return nil, false, err
	}

	config.Enabled = !config.Enabled
	if err := db.Save(&config).Error; err != nil {
		return nil, false, err
	}

	return &config, true, nil
}

// GetClient 获取LLM客户端
func (s *LLMService) GetClient(id string) (llm.LLMClient, error) {
	s.mu.RLock()
	if client, ok := s.clients[id]; ok {
		s.mu.RUnlock()
		return client, nil
	}
	s.mu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()

	if client, ok := s.clients[id]; ok {
		return client, nil
	}

	db := database.GetDB()
	var config models.LLMConfig
	if err := db.First(&config, "id = ?", id).Error; err != nil {
		return nil, nil
	}

	newClient, err := s.clientFactory.CreateClient(&config)
	if err != nil {
		return nil, err
	}

	s.clients[id] = newClient
	return newClient, nil
}
