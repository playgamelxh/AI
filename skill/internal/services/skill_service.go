package services

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/pkg/database"
	"encoding/json"
	"sync"
)

// SkillService 技能管理服务
type SkillService struct {
	mu sync.RWMutex
}

// NewSkillService 创建技能服务
func NewSkillService() *SkillService {
	return &SkillService{}
}

// ListSkills 获取所有技能
func (s *SkillService) ListSkills() ([]*models.Skill, error) {
	db := database.GetDB()
	var skills []*models.Skill
	if err := db.Find(&skills).Error; err != nil {
		return nil, err
	}

	for _, skill := range skills {
		if skill.ConfigJSON != "" {
			var config map[string]interface{}
			if err := json.Unmarshal([]byte(skill.ConfigJSON), &config); err == nil {
				skill.Config = config
			}
		}
	}

	return skills, nil
}

// GetSkill 获取单个技能
func (s *SkillService) GetSkill(id string) (*models.Skill, bool, error) {
	db := database.GetDB()
	var skill models.Skill
	err := db.First(&skill, "id = ?", id).Error
	if err != nil {
		return nil, false, err
	}

	if skill.ConfigJSON != "" {
		var config map[string]interface{}
		if err := json.Unmarshal([]byte(skill.ConfigJSON), &config); err == nil {
			skill.Config = config
		}
	}

	return &skill, true, nil
}

// CreateSkill 创建技能
func (s *SkillService) CreateSkill(skill *models.Skill) (*models.Skill, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if skill.ID == "" {
		skill.ID = skill.Name
	}

	if skill.Config != nil {
		if configJSON, err := json.Marshal(skill.Config); err == nil {
			skill.ConfigJSON = string(configJSON)
		}
	}

	db := database.GetDB()
	if err := db.Create(skill).Error; err != nil {
		return nil, err
	}

	return skill, nil
}

// UpdateSkill 更新技能
func (s *SkillService) UpdateSkill(id string, skill *models.Skill) (*models.Skill, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	var existing models.Skill
	if err := db.First(&existing, "id = ?", id).Error; err != nil {
		return nil, false, err
	}

	skill.ID = id
	if skill.Config != nil {
		if configJSON, err := json.Marshal(skill.Config); err == nil {
			skill.ConfigJSON = string(configJSON)
		}
	}

	if err := db.Save(skill).Error; err != nil {
		return nil, false, err
	}

	return skill, true, nil
}

// DeleteSkill 删除技能
func (s *SkillService) DeleteSkill(id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	result := db.Delete(&models.Skill{}, "id = ?", id)
	if result.Error != nil {
		return false, result.Error
	}
	return result.RowsAffected > 0, nil
}

// ToggleSkill 切换技能启用状态
func (s *SkillService) ToggleSkill(id string) (*models.Skill, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	db := database.GetDB()
	var skill models.Skill
	if err := db.First(&skill, "id = ?", id).Error; err != nil {
		return nil, false, err
	}

	skill.Enabled = !skill.Enabled
	if err := db.Save(&skill).Error; err != nil {
		return nil, false, err
	}

	if skill.ConfigJSON != "" {
		var config map[string]interface{}
		if err := json.Unmarshal([]byte(skill.ConfigJSON), &config); err == nil {
			skill.Config = config
		}
	}

	return &skill, true, nil
}
