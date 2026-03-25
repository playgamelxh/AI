package scheduler

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/internal/services"
	"fmt"
	"log"
	"sync"
	"time"
)

// Scheduler 任务调度器
type Scheduler struct {
	taskService   *services.TaskService
	agentService  *services.AgentService
	skillService  *services.SkillService
	llmService    *services.LLMService
	skillExecutor *services.SkillExecutor
	running       bool
	mu            sync.Mutex
	stopChan      chan struct{}
	wg            sync.WaitGroup
}

// NewScheduler 创建任务调度器
func NewScheduler(
	taskService *services.TaskService,
	agentService *services.AgentService,
	skillService *services.SkillService,
	llmService *services.LLMService,
) *Scheduler {
	return &Scheduler{
		taskService:   taskService,
		agentService:  agentService,
		skillService:  skillService,
		llmService:    llmService,
		skillExecutor: services.NewSkillExecutor(skillService, llmService),
		stopChan:      make(chan struct{}),
	}
}

// Start 启动调度器
func (s *Scheduler) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		log.Println("Scheduler is already running")
		return
	}

	s.running = true
	s.wg.Add(1)
	go s.run()

	log.Println("Scheduler started")
}

// Stop 停止调度器
func (s *Scheduler) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		log.Println("Scheduler is not running")
		return
	}

	close(s.stopChan)
	s.wg.Wait()
	s.running = false

	log.Println("Scheduler stopped")
}

// run 调度器主循环
func (s *Scheduler) run() {
	defer s.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.checkAndExecuteTasks()
		case <-s.stopChan:
			return
		}
	}
}

// checkAndExecuteTasks 检查并执行任务
func (s *Scheduler) checkAndExecuteTasks() {
	tasks, err := s.taskService.GetPendingTasks()
	if err != nil {
		log.Printf("Failed to get pending tasks: %v", err)
		return
	}

	for _, task := range tasks {
		go s.executeTask(task)
	}
}

// executeTask 执行单个任务
func (s *Scheduler) executeTask(task *models.Task) {
	log.Printf("Starting task execution: %s (%s)", task.Name, task.ID)

	if err := s.taskService.UpdateTaskStatus(task.ID, models.TaskStatusRunning, nil, ""); err != nil {
		log.Printf("Failed to update task status to running: %v", err)
		return
	}

	result, err := s.runTaskLogic(task)

	if err != nil {
		log.Printf("Task failed: %s, error: %v", task.ID, err)
		if task.RetryCount < task.MaxRetries {
			s.taskService.UpdateTaskStatus(task.ID, models.TaskStatusPending, result, err.Error())
		} else {
			s.taskService.UpdateTaskStatus(task.ID, models.TaskStatusFailed, result, err.Error())
		}
		return
	}

	log.Printf("Task completed successfully: %s", task.ID)
	s.taskService.UpdateTaskStatus(task.ID, models.TaskStatusCompleted, result, "")

	if task.Type == models.TaskTypePeriodic {
		s.scheduleNextRun(task)
	}
}

// runTaskLogic 执行任务逻辑
func (s *Scheduler) runTaskLogic(task *models.Task) (map[string]interface{}, error) {
	log.Printf("开始分析任务: %s", task.ID)

	agent, err := s.agentService.GetAgentByID(task.AgentID)
	if err != nil {
		return nil, fmt.Errorf("获取Agent失败: %w", err)
	}
	if agent == nil {
		return nil, fmt.Errorf("未找到任务对应的Agent")
	}

	result := make(map[string]interface{})
	result["task_id"] = task.ID
	result["agent_id"] = agent.ID
	result["agent_name"] = agent.Name
	result["executed_at"] = time.Now().Format(time.RFC3339)

	log.Printf("使用Agent: %s (ID: %s) 执行任务", agent.Name, agent.ID)

	analysis, err := s.skillExecutor.AnalyzeTaskRequirements(task, agent)
	if err != nil {
		return nil, fmt.Errorf("任务分析失败: %w", err)
	}

	result["analysis"] = analysis
	result["suggested_skills"] = analysis.SuggestedSkills

	if len(analysis.SuggestedSkills) > 0 {
		log.Printf("技能建议: %v", analysis.SuggestedSkills)
	}

	missingSkills := s.checkAgentSkills(agent, analysis.RequiredSkills)
	if len(missingSkills) > 0 {
		result["missing_skills"] = missingSkills
		log.Printf("Agent %s 缺少必要技能: %v", agent.Name, missingSkills)

		if len(analysis.SuggestedSkills) > 0 {
			return result, fmt.Errorf("缺少必要技能: %v。%v", missingSkills, analysis.SuggestedSkills)
		}
		return result, fmt.Errorf("缺少必要技能: %v", missingSkills)
	}

	log.Printf("Agent技能检查通过，开始执行 %d 个子任务", len(analysis.Subtasks))
	subtaskResults := make([]map[string]interface{}, 0)

	for i, subtask := range analysis.Subtasks {
		log.Printf("执行子任务 %d/%d: %s (%s)", i+1, len(analysis.Subtasks), subtask.Name, subtask.ID)

		subtaskResult, err := s.executeSubtask(agent, subtask)
		subtaskResults = append(subtaskResults, subtaskResult)

		if err != nil {
			result["subtasks"] = subtaskResults
			result["failed_subtask"] = subtask.ID
			log.Printf("子任务执行失败: %s, 错误: %v", subtask.ID, err)
			return result, fmt.Errorf("子任务 %s 执行失败: %w", subtask.ID, err)
		}

		log.Printf("子任务执行成功: %s", subtask.ID)
	}

	result["subtasks"] = subtaskResults
	result["status"] = "success"
	result["message"] = fmt.Sprintf("成功执行 %d 个子任务", len(subtaskResults))
	result["completed_subtasks"] = len(subtaskResults)

	log.Printf("任务执行完成: %s", task.ID)
	return result, nil
}

// checkAgentSkills 检查Agent是否拥有所需技能
func (s *Scheduler) checkAgentSkills(agent *models.Agent, requiredSkills map[models.SkillType]bool) []string {
	missing := make([]string, 0)

	agentSkills := make(map[models.SkillType]bool)
	agentSkillDetails := make(map[models.SkillType][]*models.Skill)

	for _, skillID := range agent.SkillIDs {
		skill, ok, err := s.skillService.GetSkill(skillID)
		if err == nil && ok && skill != nil {
			agentSkills[skill.Type] = agentSkills[skill.Type] || skill.Enabled
			agentSkillDetails[skill.Type] = append(agentSkillDetails[skill.Type], skill)

			if skill.Enabled {
				log.Printf("Agent拥有启用的技能: %s (类型: %s, ID: %s)", skill.Name, skill.Type, skill.ID)
			} else {
				log.Printf("Agent拥有禁用的技能: %s (类型: %s, ID: %s)", skill.Name, skill.Type, skill.ID)
			}
		}
	}

	for skillType := range requiredSkills {
		if !agentSkills[skillType] {
			missing = append(missing, string(skillType))

			if skills, exists := agentSkillDetails[skillType]; exists && len(skills) > 0 {
				log.Printf("Agent拥有%s类型技能但全部被禁用", skillType)
			}
		}
	}

	return missing
}

// generateSkillSuggestions 生成技能开启建议
func (s *Scheduler) generateSkillSuggestions(missingSkills []string) []string {
	suggestions := make([]string, 0)

	allSkills, err := s.skillService.ListSkills()
	if err != nil {
		log.Printf("获取技能列表失败: %v", err)
		return suggestions
	}

	for _, skillType := range missingSkills {
		foundEnabled := false
		foundDisabled := false

		for _, skill := range allSkills {
			if string(skill.Type) == skillType {
				if skill.Enabled {
					suggestion := fmt.Sprintf("建议为Agent绑定并启用技能: %s (类型: %s, ID: %s)",
						skill.Name, skill.Type, skill.ID)
					suggestions = append(suggestions, suggestion)
					foundEnabled = true
					break
				} else {
					suggestion := fmt.Sprintf("系统中有%s类型技能但已禁用: %s (ID: %s)，请先启用该技能",
						skill.Type, skill.Name, skill.ID)
					suggestions = append(suggestions, suggestion)
					foundDisabled = true
				}
			}
		}

		if !foundEnabled && !foundDisabled {
			suggestions = append(suggestions,
				fmt.Sprintf("系统中缺少%s类型的技能，请先创建相应技能", skillType))
		}
	}

	return suggestions
}

// executeSubtask 执行子任务
func (s *Scheduler) executeSubtask(agent *models.Agent, subtask *services.Subtask) (map[string]interface{}, error) {
	log.Printf("开始执行子任务: %s (%s)", subtask.Name, subtask.ID)
	startTime := time.Now()

	result := map[string]interface{}{
		"subtask_id":   subtask.ID,
		"subtask_name": subtask.Name,
		"skill_type":   subtask.SkillType,
		"description":  subtask.Description,
		"started_at":   startTime.Format(time.RFC3339),
	}

	log.Printf("正在为子任务寻找匹配的技能，所需技能类型: %s", subtask.SkillType)

	var skillToUse *models.Skill
	var availableSkills []*models.Skill

	for _, skillID := range agent.SkillIDs {
		skill, ok, err := s.skillService.GetSkill(skillID)
		if err == nil && ok && skill != nil {
			if skill.Type == subtask.SkillType {
				if skill.Enabled {
					skillToUse = skill
					log.Printf("找到匹配的启用技能: %s (ID: %s)", skill.Name, skill.ID)
					break
				} else {
					availableSkills = append(availableSkills, skill)
					log.Printf("找到匹配的禁用技能: %s (ID: %s)", skill.Name, skill.ID)
				}
			}
		}
	}

	if skillToUse == nil {
		result["success"] = false
		result["duration_ms"] = time.Since(startTime).Milliseconds()
		result["completed_at"] = time.Now().Format(time.RFC3339)

		if len(availableSkills) > 0 {
			skillNames := make([]string, 0, len(availableSkills))
			for _, s := range availableSkills {
				skillNames = append(skillNames, fmt.Sprintf("%s (ID: %s)", s.Name, s.ID))
			}
			errorMsg := fmt.Sprintf("Agent有%s类型的技能但全部被禁用: %v", subtask.SkillType, skillNames)
			result["error"] = errorMsg
			log.Printf(errorMsg)
			return result, fmt.Errorf(errorMsg)
		}

		errorMsg := fmt.Sprintf("Agent没有可用的%s类型技能", subtask.SkillType)
		result["error"] = errorMsg
		log.Printf(errorMsg)
		return result, fmt.Errorf(errorMsg)
	}

	result["skill_used"] = skillToUse.Name
	result["skill_id"] = skillToUse.ID
	result["skill_config"] = skillToUse.Config

	log.Printf("使用技能: %s (ID: %s) 执行子任务", skillToUse.Name, skillToUse.ID)
	log.Printf("子任务参数: %+v", subtask.Params)

	skillResult, err := s.skillExecutor.ExecuteSkill(skillToUse, subtask.Params)

	duration := time.Since(startTime)
	result["duration_ms"] = duration.Milliseconds()
	result["completed_at"] = time.Now().Format(time.RFC3339)

	if err != nil {
		result["error"] = err.Error()
		result["success"] = false
		result["skill_result"] = skillResult
		subtask.Error = err.Error()
		subtask.Completed = false

		log.Printf("子任务执行失败: %s, 耗时: %v, 错误: %v", subtask.ID, duration, err)
		return result, err
	}

	result["skill_result"] = skillResult
	result["success"] = true
	subtask.Completed = true
	subtask.Result = skillResult

	log.Printf("子任务执行成功: %s, 耗时: %v", subtask.ID, duration)
	return result, nil
}

// scheduleNextRun 调度周期性任务的下次运行
func (s *Scheduler) scheduleNextRun(task *models.Task) {
	now := time.Now()
	nextTime := now.Add(1 * time.Hour)
	task.NextRunTime = &nextTime

	if err := s.taskService.UpdateTask(task); err != nil {
		log.Printf("Failed to schedule next run for task %s: %v", task.ID, err)
	}
}
