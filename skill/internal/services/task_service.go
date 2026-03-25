package services

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/pkg/database"
	"encoding/json"
	"errors"
	"sync"
	"time"

	"gorm.io/gorm"
)

// TaskService Task管理服务
type TaskService struct {
	db *gorm.DB
	mu sync.RWMutex
}

// NewTaskService 创建Task服务
func NewTaskService() *TaskService {
	return &TaskService{
		db: database.GetDB(),
	}
}

// GetAllTasks 获取所有任务
func (s *TaskService) GetAllTasks() ([]*models.Task, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var tasks []*models.Task
	if err := s.db.Preload("Agent").Find(&tasks).Error; err != nil {
		return nil, err
	}

	for _, task := range tasks {
		if err := s.loadTaskJSONFields(task); err != nil {
			return nil, err
		}
	}

	return tasks, nil
}

// GetTasksByAgentID 根据AgentID获取任务
func (s *TaskService) GetTasksByAgentID(agentID string) ([]*models.Task, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var tasks []*models.Task
	if err := s.db.Preload("Agent").Where("agent_id = ?", agentID).Find(&tasks).Error; err != nil {
		return nil, err
	}

	for _, task := range tasks {
		if err := s.loadTaskJSONFields(task); err != nil {
			return nil, err
		}
	}

	return tasks, nil
}

// GetTaskByID 根据ID获取任务
func (s *TaskService) GetTaskByID(id string) (*models.Task, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var task models.Task
	if err := s.db.Preload("Agent").Where("id = ?", id).First(&task).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, nil
		}
		return nil, err
	}

	if err := s.loadTaskJSONFields(&task); err != nil {
		return nil, err
	}

	return &task, nil
}

// CreateTask 创建任务
func (s *TaskService) CreateTask(task *models.Task) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.saveTaskJSONFields(task); err != nil {
		return err
	}

	if task.Type == models.TaskTypeOnce && task.ScheduledTime == nil {
		now := time.Now()
		task.ScheduledTime = &now
	}

	return s.db.Create(task).Error
}

// UpdateTask 更新任务
func (s *TaskService) UpdateTask(task *models.Task) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.saveTaskJSONFields(task); err != nil {
		return err
	}

	task.UpdatedAt = time.Now()
	return s.db.Save(task).Error
}

// DeleteTask 删除任务
func (s *TaskService) DeleteTask(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.db.Delete(&models.Task{}, "id = ?", id).Error
}

// ToggleTask 切换任务启用状态
func (s *TaskService) ToggleTask(id string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	task, err := s.GetTaskByID(id)
	if err != nil {
		return false, err
	}
	if task == nil {
		return false, errors.New("task not found")
	}

	task.Enabled = !task.Enabled
	if err := s.UpdateTask(task); err != nil {
		return false, err
	}

	return task.Enabled, nil
}

// UpdateTaskStatus 更新任务状态
func (s *TaskService) UpdateTaskStatus(id string, status models.TaskStatus, result map[string]interface{}, errMsg string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	task, err := s.GetTaskByID(id)
	if err != nil {
		return err
	}
	if task == nil {
		return errors.New("task not found")
	}

	task.Status = status
	now := time.Now()
	task.LastRunTime = &now

	if result != nil {
		task.Result = result
	}

	if errMsg != "" {
		task.ErrorMessage = errMsg
	}

	if status == models.TaskStatusFailed {
		task.RetryCount++
	}

	return s.UpdateTask(task)
}

// GetPendingTasks 获取待执行的任务
func (s *TaskService) GetPendingTasks() ([]*models.Task, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var tasks []*models.Task
	now := time.Now()
	
	query := s.db.Preload("Agent").Where("status = ? AND enabled = ?", models.TaskStatusPending, true)
	query = query.Where("(scheduled_time IS NULL OR scheduled_time <= ?) OR (next_run_time IS NOT NULL AND next_run_time <= ?)", now, now)
	
	if err := query.Find(&tasks).Error; err != nil {
		return nil, err
	}

	for _, task := range tasks {
		if err := s.loadTaskJSONFields(task); err != nil {
			return nil, err
		}
	}

	return tasks, nil
}

// loadTaskJSONFields 加载Task的JSON字段
func (s *TaskService) loadTaskJSONFields(task *models.Task) error {
	if task.ContentJSON != "" {
		if err := json.Unmarshal([]byte(task.ContentJSON), &task.Content); err != nil {
			return err
		}
	}
	if task.Content == nil {
		task.Content = make(map[string]interface{})
	}

	if task.ResultJSON != "" {
		if err := json.Unmarshal([]byte(task.ResultJSON), &task.Result); err != nil {
			return err
		}
	}
	if task.Result == nil {
		task.Result = make(map[string]interface{})
	}

	return nil
}

// saveTaskJSONFields 保存Task的JSON字段
func (s *TaskService) saveTaskJSONFields(task *models.Task) error {
	if task.Content != nil {
		data, err := json.Marshal(task.Content)
		if err != nil {
			return err
		}
		task.ContentJSON = string(data)
	}

	if task.Result != nil {
		data, err := json.Marshal(task.Result)
		if err != nil {
			return err
		}
		task.ResultJSON = string(data)
	}

	return nil
}
