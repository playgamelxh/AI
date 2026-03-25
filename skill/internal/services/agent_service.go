package services

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/pkg/database"
	"encoding/json"
	"errors"
	"sync"

	"gorm.io/gorm"
)

// AgentService Agent管理服务
type AgentService struct {
	db *gorm.DB
	mu sync.RWMutex
}

// NewAgentService 创建Agent服务
func NewAgentService() *AgentService {
	return &AgentService{
		db: database.GetDB(),
	}
}

// GetAllAgents 获取所有Agent
func (s *AgentService) GetAllAgents() ([]*models.Agent, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var agents []*models.Agent
	if err := s.db.Preload("LLMConfig").Find(&agents).Error; err != nil {
		return nil, err
	}

	for _, agent := range agents {
		if err := s.loadAgentJSONFields(agent); err != nil {
			return nil, err
		}
	}

	return agents, nil
}

// GetAgentByID 根据ID获取Agent
func (s *AgentService) GetAgentByID(id string) (*models.Agent, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var agent models.Agent
	if err := s.db.Preload("LLMConfig").Where("id = ?", id).First(&agent).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}

	if err := s.loadAgentJSONFields(&agent); err != nil {
		return nil, err
	}

	return &agent, nil
}

// CreateAgent 创建Agent
func (s *AgentService) CreateAgent(agent *models.Agent) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.saveAgentJSONFields(agent); err != nil {
		return err
	}

	return s.db.Create(agent).Error
}

// UpdateAgent 更新Agent
func (s *AgentService) UpdateAgent(agent *models.Agent) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.saveAgentJSONFields(agent); err != nil {
		return err
	}

	return s.db.Save(agent).Error
}

// DeleteAgent 删除Agent
func (s *AgentService) DeleteAgent(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.db.Delete(&models.Agent{}, "id = ?", id).Error
}

// ToggleAgent 切换Agent启用状态
func (s *AgentService) ToggleAgent(id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	agent, err := s.GetAgentByID(id)
	if err != nil {
		return false, err
	}
	if agent == nil {
		return false, errors.New("agent not found")
	}

	agent.Enabled = !agent.Enabled
	if err := s.UpdateAgent(agent); err != nil {
		return false, err
	}

	return agent.Enabled, nil
}

// AddSkillToAgent 给Agent添加技能
func (s *AgentService) AddSkillToAgent(agentID, skillID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	agent, err := s.GetAgentByID(agentID)
	if err != nil {
		return err
	}
	if agent == nil {
		return errors.New("agent not found")
	}

	for _, id := range agent.SkillIDs {
		if id == skillID {
			return nil
		}
	}

	agent.SkillIDs = append(agent.SkillIDs, skillID)
	return s.UpdateAgent(agent)
}

// RemoveSkillFromAgent 从Agent移除技能
func (s *AgentService) RemoveSkillFromAgent(agentID, skillID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	agent, err := s.GetAgentByID(agentID)
	if err != nil {
		return err
	}
	if agent == nil {
		return errors.New("agent not found")
	}

	var newSkillIDs []string
	for _, id := range agent.SkillIDs {
		if id != skillID {
			newSkillIDs = append(newSkillIDs, id)
		}
	}

	agent.SkillIDs = newSkillIDs
	return s.UpdateAgent(agent)
}

// loadAgentJSONFields 加载Agent的JSON字段
func (s *AgentService) loadAgentJSONFields(agent *models.Agent) error {
	if agent.SkillIDsJSON != "" {
		if err := json.Unmarshal([]byte(agent.SkillIDsJSON), &agent.SkillIDs); err != nil {
			return err
		}
	}
	if agent.SkillIDs == nil {
		agent.SkillIDs = []string{}
	}
	return nil
}

// saveAgentJSONFields 保存Agent的JSON字段
func (s *AgentService) saveAgentJSONFields(agent *models.Agent) error {
	if agent.SkillIDs != nil {
		data, err := json.Marshal(agent.SkillIDs)
		if err != nil {
			return err
		}
		agent.SkillIDsJSON = string(data)
	}
	return nil
}
