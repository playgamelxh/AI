package api

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/internal/services"
	"net/http"

	"github.com/gin-gonic/gin"
)

// Handler API处理器
type Handler struct {
	llmService   *services.LLMService
	skillService *services.SkillService
	agentService *services.AgentService
	taskService  *services.TaskService
}

// NewHandler 创建API处理器
func NewHandler(llmService *services.LLMService, skillService *services.SkillService, agentService *services.AgentService, taskService *services.TaskService) *Handler {
	return &Handler{
		llmService:   llmService,
		skillService: skillService,
		agentService: agentService,
		taskService:  taskService,
	}
}

// Health 健康检查
func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, models.NewSuccessResponse(gin.H{
		"status": "ok",
	}))
}

// ListLLMConfigs 获取所有LLM配置
func (h *Handler) ListLLMConfigs(c *gin.Context) {
	configs, err := h.llmService.ListConfigs()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(configs))
}

// GetLLMConfig 获取单个LLM配置
func (h *Handler) GetLLMConfig(c *gin.Context) {
	id := c.Param("id")
	config, ok, err := h.llmService.GetConfig(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !ok {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("配置不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(config))
}

// CreateLLMConfig 创建LLM配置
func (h *Handler) CreateLLMConfig(c *gin.Context) {
	var config models.LLMConfig
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	created, err := h.llmService.CreateConfig(&config)
	if err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusCreated, models.NewSuccessResponse(created))
}

// UpdateLLMConfig 更新LLM配置
func (h *Handler) UpdateLLMConfig(c *gin.Context) {
	id := c.Param("id")
	var config models.LLMConfig
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	updated, err := h.llmService.UpdateConfig(id, &config)
	if err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	if updated == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("配置不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(updated))
}

// DeleteLLMConfig 删除LLM配置
func (h *Handler) DeleteLLMConfig(c *gin.Context) {
	id := c.Param("id")
	deleted, err := h.llmService.DeleteConfig(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !deleted {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("配置不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("删除成功", nil))
}

// ToggleLLMConfig 切换LLM配置启用状态
func (h *Handler) ToggleLLMConfig(c *gin.Context) {
	id := c.Param("id")
	config, ok, err := h.llmService.ToggleConfig(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !ok {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("配置不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(config))
}

// Chat 聊天接口
func (h *Handler) Chat(c *gin.Context) {
	var req models.ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	client, err := h.llmService.GetClient(req.ConfigID)
	if err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	if client == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("配置不存在"))
		return
	}

	resp, err := client.Chat(c.Request.Context(), req.Messages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}

	c.JSON(http.StatusOK, models.NewSuccessResponse(resp))
}

// ListSkills 获取所有技能
func (h *Handler) ListSkills(c *gin.Context) {
	skills, err := h.skillService.ListSkills()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(skills))
}

// GetSkill 获取单个技能
func (h *Handler) GetSkill(c *gin.Context) {
	id := c.Param("id")
	skill, ok, err := h.skillService.GetSkill(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !ok {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("技能不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(skill))
}

// CreateSkill 创建技能
func (h *Handler) CreateSkill(c *gin.Context) {
	var skill models.Skill
	if err := c.ShouldBindJSON(&skill); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	created, err := h.skillService.CreateSkill(&skill)
	if err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusCreated, models.NewSuccessResponse(created))
}

// UpdateSkill 更新技能
func (h *Handler) UpdateSkill(c *gin.Context) {
	id := c.Param("id")
	var skill models.Skill
	if err := c.ShouldBindJSON(&skill); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	updated, ok, err := h.skillService.UpdateSkill(id, &skill)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !ok {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("技能不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(updated))
}

// DeleteSkill 删除技能
func (h *Handler) DeleteSkill(c *gin.Context) {
	id := c.Param("id")
	deleted, err := h.skillService.DeleteSkill(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !deleted {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("技能不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("删除成功", nil))
}

// ToggleSkill 切换技能启用状态
func (h *Handler) ToggleSkill(c *gin.Context) {
	id := c.Param("id")
	skill, ok, err := h.skillService.ToggleSkill(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if !ok {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("技能不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(skill))
}

// ListAgents 获取所有Agent
func (h *Handler) ListAgents(c *gin.Context) {
	agents, err := h.agentService.GetAllAgents()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(agents))
}

// GetAgent 获取单个Agent
func (h *Handler) GetAgent(c *gin.Context) {
	id := c.Param("id")
	agent, err := h.agentService.GetAgentByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("Agent不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(agent))
}

// CreateAgent 创建Agent
func (h *Handler) CreateAgent(c *gin.Context) {
	var agent models.Agent
	if err := c.ShouldBindJSON(&agent); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	if agent.ID == "" {
		newAgent := models.NewAgent(agent.Name, agent.Description, agent.LLMConfigID)
		newAgent.SkillIDs = agent.SkillIDs
		agent = *newAgent
	}

	if err := h.agentService.CreateAgent(&agent); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusCreated, models.NewSuccessResponse(agent))
}

// UpdateAgent 更新Agent
func (h *Handler) UpdateAgent(c *gin.Context) {
	id := c.Param("id")
	var agent models.Agent
	if err := c.ShouldBindJSON(&agent); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	existing, err := h.agentService.GetAgentByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if existing == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("Agent不存在"))
		return
	}

	agent.ID = id
	if err := h.agentService.UpdateAgent(&agent); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(agent))
}

// DeleteAgent 删除Agent
func (h *Handler) DeleteAgent(c *gin.Context) {
	id := c.Param("id")
	if err := h.agentService.DeleteAgent(id); err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("删除成功", nil))
}

// ToggleAgent 切换Agent启用状态
func (h *Handler) ToggleAgent(c *gin.Context) {
	id := c.Param("id")
	enabled, err := h.agentService.ToggleAgent(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(gin.H{"enabled": enabled}))
}

// AddSkillToAgent 给Agent添加技能
func (h *Handler) AddSkillToAgent(c *gin.Context) {
	agentID := c.Param("id")
	var req struct {
		SkillID string `json:"skill_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	if err := h.agentService.AddSkillToAgent(agentID, req.SkillID); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("技能添加成功", nil))
}

// RemoveSkillFromAgent 从Agent移除技能
func (h *Handler) RemoveSkillFromAgent(c *gin.Context) {
	agentID := c.Param("id")
	var req struct {
		SkillID string `json:"skill_id"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	if err := h.agentService.RemoveSkillFromAgent(agentID, req.SkillID); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("技能移除成功", nil))
}

// ListTasks 获取所有任务
func (h *Handler) ListTasks(c *gin.Context) {
	tasks, err := h.taskService.GetAllTasks()
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(tasks))
}

// GetTasksByAgent 获取Agent的任务
func (h *Handler) GetTasksByAgent(c *gin.Context) {
	agentID := c.Param("agent_id")
	tasks, err := h.taskService.GetTasksByAgentID(agentID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(tasks))
}

// GetTask 获取单个任务
func (h *Handler) GetTask(c *gin.Context) {
	id := c.Param("id")
	task, err := h.taskService.GetTaskByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if task == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("任务不存在"))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(task))
}

// CreateTask 创建任务
func (h *Handler) CreateTask(c *gin.Context) {
	var task models.Task
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	if task.ID == "" {
		newTask := models.NewTask(task.AgentID, task.Name, task.Description, task.Type)
		newTask.CronExpr = task.CronExpr
		newTask.ScheduledTime = task.ScheduledTime
		newTask.Content = task.Content
		newTask.MaxRetries = task.MaxRetries
		task = *newTask
	}

	if err := h.taskService.CreateTask(&task); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusCreated, models.NewSuccessResponse(task))
}

// UpdateTask 更新任务
func (h *Handler) UpdateTask(c *gin.Context) {
	id := c.Param("id")
	var task models.Task
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse("无效的请求参数"))
		return
	}

	existing, err := h.taskService.GetTaskByID(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	if existing == nil {
		c.JSON(http.StatusNotFound, models.NewErrorResponse("任务不存在"))
		return
	}

	task.ID = id
	if err := h.taskService.UpdateTask(&task); err != nil {
		c.JSON(http.StatusBadRequest, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(task))
}

// DeleteTask 删除任务
func (h *Handler) DeleteTask(c *gin.Context) {
	id := c.Param("id")
	if err := h.taskService.DeleteTask(id); err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewMessageResponse("删除成功", nil))
}

// ToggleTask 切换任务启用状态
func (h *Handler) ToggleTask(c *gin.Context) {
	id := c.Param("id")
	enabled, err := h.taskService.ToggleTask(id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, models.NewErrorResponse(err.Error()))
		return
	}
	c.JSON(http.StatusOK, models.NewSuccessResponse(gin.H{"enabled": enabled}))
}
