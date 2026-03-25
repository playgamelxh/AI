package services

import (
	"ai-agent-skill/internal/models"
	"ai-agent-skill/pkg/llm"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// TaskAnalysis 任务分析结果
type TaskAnalysis struct {
	TaskID          string                    `json:"task_id"`
	RequiredSkills  map[models.SkillType]bool `json:"required_skills"`
	Subtasks        []*Subtask                `json:"subtasks"`
	SuggestedSkills []string                  `json:"suggested_skills"`
}

// Subtask 子任务
type Subtask struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	SkillType   models.SkillType       `json:"skill_type"`
	Params      map[string]interface{} `json:"params"`
	Result      map[string]interface{} `json:"result,omitempty"`
	Error       string                 `json:"error,omitempty"`
	Completed   bool                   `json:"completed"`
}

// SkillExecutor 技能执行器
type SkillExecutor struct {
	skillService *SkillService
	llmService   *LLMService
}

// NewSkillExecutor 创建技能执行器
func NewSkillExecutor(skillService *SkillService, llmService *LLMService) *SkillExecutor {
	return &SkillExecutor{
		skillService: skillService,
		llmService:   llmService,
	}
}

// ExecuteSkill 执行技能
func (e *SkillExecutor) ExecuteSkill(skill *models.Skill, params map[string]interface{}) (map[string]interface{}, error) {
	if !skill.Enabled {
		return nil, fmt.Errorf("skill %s is disabled", skill.Name)
	}

	switch skill.Type {
	case models.SkillTypeCommand:
		return e.executeCommandSkill(skill, params)
	case models.SkillTypeFile:
		return e.executeFileSkill(skill, params)
	case models.SkillTypeApplication:
		return e.executeApplicationSkill(skill, params)
	default:
		return nil, fmt.Errorf("unknown skill type: %s", skill.Type)
	}
}

// executeCommandSkill 执行命令技能
func (e *SkillExecutor) executeCommandSkill(skill *models.Skill, params map[string]interface{}) (map[string]interface{}, error) {
	command, ok := params["command"].(string)
	if !ok {
		return nil, fmt.Errorf("command parameter is required")
	}

	log.Printf("Executing command skill: %s, command: %s", skill.Name, command)

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/C", command)
	} else {
		cmd = exec.Command("bash", "-c", command)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	result := map[string]interface{}{
		"skill_type": models.SkillTypeCommand,
		"skill_name": skill.Name,
		"command":    command,
		"stdout":     stdout.String(),
		"stderr":     stderr.String(),
		"exit_code":  cmd.ProcessState.ExitCode(),
		"success":    err == nil,
	}

	if err != nil {
		result["error"] = err.Error()
		return result, err
	}

	return result, nil
}

// executeFileSkill 执行文件技能
func (e *SkillExecutor) executeFileSkill(skill *models.Skill, params map[string]interface{}) (map[string]interface{}, error) {
	action, ok := params["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action parameter is required (read, write, delete, list)")
	}

	log.Printf("Executing file skill: %s, action: %s", skill.Name, action)

	result := map[string]interface{}{
		"skill_type": models.SkillTypeFile,
		"skill_name": skill.Name,
		"action":     action,
	}

	switch action {
	case "list":
		return e.handleFileList(params, result)
	case "read":
		return e.handleFileRead(params, result)
	case "write":
		return e.handleFileWrite(params, result)
	case "delete":
		return e.handleFileDelete(params, result)
	default:
		return nil, fmt.Errorf("unknown file action: %s", action)
	}
}

// handleFileList 处理文件列表操作
func (e *SkillExecutor) handleFileList(params map[string]interface{}, result map[string]interface{}) (map[string]interface{}, error) {
	path, ok := params["path"].(string)
	if !ok {
		path = "."
	}

	cmd := exec.Command("ls", "-la", path)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	result["path"] = path
	result["stdout"] = stdout.String()
	result["stderr"] = stderr.String()
	result["success"] = err == nil

	if err != nil {
		result["error"] = err.Error()
		return result, err
	}

	return result, nil
}

// handleFileRead 处理文件读取操作
func (e *SkillExecutor) handleFileRead(params map[string]interface{}, result map[string]interface{}) (map[string]interface{}, error) {
	path, ok := params["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path parameter is required")
	}

	cmd := exec.Command("cat", path)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	result["path"] = path
	result["content"] = stdout.String()
	result["stderr"] = stderr.String()
	result["success"] = err == nil

	if err != nil {
		result["error"] = err.Error()
		return result, err
	}

	return result, nil
}

// handleFileWrite 处理文件写入操作
func (e *SkillExecutor) handleFileWrite(params map[string]interface{}, result map[string]interface{}) (map[string]interface{}, error) {
	path, ok := params["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path parameter is required")
	}

	content, ok := params["content"].(string)
	if !ok {
		return nil, fmt.Errorf("content parameter is required")
	}

	cmd := exec.Command("bash", "-c", fmt.Sprintf("cat > %s << 'EOF'\n%s\nEOF", path, content))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	result["path"] = path
	result["stderr"] = stderr.String()
	result["success"] = err == nil

	if err != nil {
		result["error"] = err.Error()
		return result, err
	}

	return result, nil
}

// handleFileDelete 处理文件删除操作
func (e *SkillExecutor) handleFileDelete(params map[string]interface{}, result map[string]interface{}) (map[string]interface{}, error) {
	path, ok := params["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path parameter is required")
	}

	cmd := exec.Command("rm", "-f", path)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Run()
	result["path"] = path
	result["stderr"] = stderr.String()
	result["success"] = err == nil

	if err != nil {
		result["error"] = err.Error()
		return result, err
	}

	return result, nil
}

// executeApplicationSkill 执行应用程序技能
func (e *SkillExecutor) executeApplicationSkill(skill *models.Skill, params map[string]interface{}) (map[string]interface{}, error) {
	action, ok := params["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action parameter is required (open_browser, open_file, open_app)")
	}

	log.Printf("Executing application skill: %s, action: %s", skill.Name, action)

	result := map[string]interface{}{
		"skill_type": models.SkillTypeApplication,
		"skill_name": skill.Name,
		"action":     action,
	}

	var cmd *exec.Cmd
	switch action {
	case "open_browser":
		url, ok := params["url"].(string)
		if !ok {
			return nil, fmt.Errorf("url parameter is required")
		}
		result["url"] = url
		if runtime.GOOS == "darwin" {
			cmd = exec.Command("open", url)
		} else if runtime.GOOS == "windows" {
			cmd = exec.Command("cmd", "/c", "start", url)
		} else {
			cmd = exec.Command("xdg-open", url)
		}
	case "open_file":
		path, ok := params["path"].(string)
		if !ok {
			return nil, fmt.Errorf("path parameter is required")
		}
		result["path"] = path
		if runtime.GOOS == "darwin" {
			cmd = exec.Command("open", path)
		} else if runtime.GOOS == "windows" {
			cmd = exec.Command("cmd", "/c", "start", "", path)
		} else {
			cmd = exec.Command("xdg-open", path)
		}
	case "open_app":
		app, ok := params["app"].(string)
		if !ok {
			return nil, fmt.Errorf("app parameter is required")
		}
		result["app"] = app
		if runtime.GOOS == "darwin" {
			cmd = exec.Command("open", "-a", app)
		} else if runtime.GOOS == "windows" {
			cmd = exec.Command("cmd", "/c", "start", app)
		} else {
			cmd = exec.Command(app)
		}
	default:
		return nil, fmt.Errorf("unknown application action: %s", action)
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err := cmd.Start()
	result["success"] = err == nil

	if err != nil {
		result["error"] = err.Error()
		result["stderr"] = stderr.String()
		return result, err
	}

	go cmd.Wait()
	return result, nil
}

// AnalyzeTaskRequirements 分析任务需求
func (e *SkillExecutor) AnalyzeTaskRequirements(task *models.Task, agent *models.Agent) (*TaskAnalysis, error) {
	analysis := &TaskAnalysis{
		TaskID:          task.ID,
		RequiredSkills:  make(map[models.SkillType]bool),
		Subtasks:        make([]*Subtask, 0),
		SuggestedSkills: make([]string, 0),
	}

	log.Printf("开始分析任务: %s, Agent: %s", task.ID, agent.Name)

	allSkills, err := e.skillService.ListSkills()
	if err != nil {
		log.Printf("获取技能列表失败: %v", err)
	}

	agentSkills := e.getAgentSkills(agent)
	log.Printf("Agent已绑定的技能类型: %v", e.getSkillTypes(agentSkills))

	taskInfo := e.extractTaskInfo(task)
	log.Printf("提取的任务信息: %+v", taskInfo)

	analysis.RequiredSkills = e.determineRequiredSkills(taskInfo)
	log.Printf("任务所需技能: %v", e.getSkillTypesFromMap(analysis.RequiredSkills))

	analysis.Subtasks = e.generateSubtasks(task, taskInfo, analysis.RequiredSkills)
	log.Printf("生成的子任务数量: %d", len(analysis.Subtasks))

	analysis.SuggestedSkills = e.generateSkillSuggestions(analysis.RequiredSkills, agentSkills, allSkills)
	log.Printf("技能建议: %v", analysis.SuggestedSkills)

	return analysis, nil
}

// extractTaskInfo 提取任务信息
func (e *SkillExecutor) extractTaskInfo(task *models.Task) map[string]interface{} {
	info := make(map[string]interface{})

	contentJSON, err := json.Marshal(task.Content)
	if err != nil {
		return info
	}
	contentStr := string(contentJSON)

	fullText := task.Name + " " + task.Description + " " + contentStr
	fullTextLower := strings.ToLower(fullText)

	info["has_command"] = strings.Contains(fullTextLower, "command") || strings.Contains(fullTextLower, "execute") ||
		strings.Contains(fullTextLower, "run") || strings.Contains(fullTextLower, "执行") ||
		strings.Contains(fullTextLower, "运行") || strings.Contains(fullTextLower, "脚本")
	info["has_file"] = strings.Contains(fullTextLower, "file") || strings.Contains(fullTextLower, "read") ||
		strings.Contains(fullTextLower, "write") || strings.Contains(fullTextLower, "delete") ||
		strings.Contains(fullTextLower, "list") || strings.Contains(fullTextLower, "path") ||
		strings.Contains(fullTextLower, "文件") || strings.Contains(fullTextLower, "读取") ||
		strings.Contains(fullTextLower, "写入") || strings.Contains(fullTextLower, "目录")
	info["has_application"] = strings.Contains(fullTextLower, "browser") || strings.Contains(fullTextLower, "open") ||
		strings.Contains(fullTextLower, "app") || strings.Contains(fullTextLower, "application") ||
		strings.Contains(fullTextLower, "url") || strings.Contains(fullTextLower, "浏览器") ||
		strings.Contains(fullTextLower, "打开") || strings.Contains(fullTextLower, "应用")

	info["content"] = task.Content
	info["description"] = task.Description
	info["name"] = task.Name

	return info
}

// determineRequiredSkills 确定所需技能
func (e *SkillExecutor) determineRequiredSkills(taskInfo map[string]interface{}) map[models.SkillType]bool {
	required := make(map[models.SkillType]bool)

	if hasCmd, ok := taskInfo["has_command"].(bool); ok && hasCmd {
		required[models.SkillTypeCommand] = true
	}
	if hasFile, ok := taskInfo["has_file"].(bool); ok && hasFile {
		required[models.SkillTypeFile] = true
	}
	if hasApp, ok := taskInfo["has_application"].(bool); ok && hasApp {
		required[models.SkillTypeApplication] = true
	}

	if len(required) == 0 {
		required[models.SkillTypeCommand] = true
	}

	return required
}

// generateSubtasks 生成子任务
func (e *SkillExecutor) generateSubtasks(task *models.Task, taskInfo map[string]interface{}, requiredSkills map[models.SkillType]bool) []*Subtask {
	subtasks := make([]*Subtask, 0)
	content := task.Content

	if requiredSkills[models.SkillTypeCommand] {
		if command, ok := content["command"].(string); ok && command != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-cmd-%d", task.ID, len(subtasks)+1),
				Name:        "执行命令",
				Description: fmt.Sprintf("执行系统命令: %s", command),
				SkillType:   models.SkillTypeCommand,
				Params: map[string]interface{}{
					"command": command,
				},
			})
		} else if script, ok := content["script"].(string); ok && script != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-cmd-%d", task.ID, len(subtasks)+1),
				Name:        "执行脚本",
				Description: fmt.Sprintf("执行脚本: %s", script),
				SkillType:   models.SkillTypeCommand,
				Params: map[string]interface{}{
					"command": script,
				},
			})
		}
	}

	if requiredSkills[models.SkillTypeFile] {
		if fileAction, ok := content["file_action"].(string); ok && fileAction != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-file-%d", task.ID, len(subtasks)+1),
				Name:        "文件操作",
				Description: fmt.Sprintf("执行文件操作: %s", fileAction),
				SkillType:   models.SkillTypeFile,
				Params:      content,
			})
		} else if path, ok := content["path"].(string); ok && path != "" {
			action := "list"
			if content["action"] != nil {
				action = content["action"].(string)
			}
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-file-%d", task.ID, len(subtasks)+1),
				Name:        fmt.Sprintf("文件%s操作", action),
				Description: fmt.Sprintf("对路径 %s 执行 %s 操作", path, action),
				SkillType:   models.SkillTypeFile,
				Params:      content,
			})
		}
	}

	if requiredSkills[models.SkillTypeApplication] {
		if appAction, ok := content["app_action"].(string); ok && appAction != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-app-%d", task.ID, len(subtasks)+1),
				Name:        "应用程序操作",
				Description: fmt.Sprintf("执行应用程序操作: %s", appAction),
				SkillType:   models.SkillTypeApplication,
				Params:      content,
			})
		} else if url, ok := content["url"].(string); ok && url != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-app-%d", task.ID, len(subtasks)+1),
				Name:        "打开浏览器",
				Description: fmt.Sprintf("在浏览器中打开: %s", url),
				SkillType:   models.SkillTypeApplication,
				Params: map[string]interface{}{
					"action": "open_browser",
					"url":    url,
				},
			})
		} else if app, ok := content["app"].(string); ok && app != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          fmt.Sprintf("%s-app-%d", task.ID, len(subtasks)+1),
				Name:        "打开应用程序",
				Description: fmt.Sprintf("打开应用程序: %s", app),
				SkillType:   models.SkillTypeApplication,
				Params: map[string]interface{}{
					"action": "open_app",
					"app":    app,
				},
			})
		}
	}

	if len(subtasks) == 0 {
		subtasks = append(subtasks, &Subtask{
			ID:          fmt.Sprintf("%s-default", task.ID),
			Name:        "执行默认任务",
			Description: task.Description,
			SkillType:   models.SkillTypeCommand,
			Params:      content,
		})
	}

	return subtasks
}

// getAgentSkills 获取Agent的技能
func (e *SkillExecutor) getAgentSkills(agent *models.Agent) []*models.Skill {
	skills := make([]*models.Skill, 0)
	for _, skillID := range agent.SkillIDs {
		skill, ok, err := e.skillService.GetSkill(skillID)
		if err == nil && ok && skill != nil && skill.Enabled {
			skills = append(skills, skill)
		}
	}
	return skills
}

// getSkillTypes 获取技能类型列表
func (e *SkillExecutor) getSkillTypes(skills []*models.Skill) []string {
	types := make([]string, 0)
	seen := make(map[models.SkillType]bool)
	for _, skill := range skills {
		if !seen[skill.Type] {
			seen[skill.Type] = true
			types = append(types, string(skill.Type))
		}
	}
	return types
}

// getSkillTypesFromMap 从map获取技能类型列表
func (e *SkillExecutor) getSkillTypesFromMap(skillMap map[models.SkillType]bool) []string {
	types := make([]string, 0)
	for skillType := range skillMap {
		types = append(types, string(skillType))
	}
	return types
}

// generateSkillSuggestions 生成技能建议
func (e *SkillExecutor) generateSkillSuggestions(requiredSkills map[models.SkillType]bool, agentSkills []*models.Skill, allSkills []*models.Skill) []string {
	suggestions := make([]string, 0)

	agentSkillTypes := make(map[models.SkillType]bool)
	for _, skill := range agentSkills {
		agentSkillTypes[skill.Type] = true
	}

	for requiredType := range requiredSkills {
		if !agentSkillTypes[requiredType] {
			found := false
			for _, skill := range allSkills {
				if skill.Type == requiredType && skill.Enabled {
					suggestions = append(suggestions,
						fmt.Sprintf("建议为Agent启用技能: %s (类型: %s, ID: %s)",
							skill.Name, skill.Type, skill.ID))
					found = true
					break
				}
			}
			if !found {
				suggestions = append(suggestions,
					fmt.Sprintf("系统中缺少%s类型的启用技能，请先创建或启用相应技能", requiredType))
			}
		}
	}

	return suggestions
}

// AnalyzeWithLLM 使用LLM进行智能任务分析（可选功能）
func (e *SkillExecutor) AnalyzeWithLLM(ctx context.Context, task *models.Task, agent *models.Agent) (*TaskAnalysis, error) {
	if e.llmService == nil || agent.LLMConfigID == "" {
		log.Println("LLM服务不可用或Agent未配置LLM，使用基础分析")
		return e.AnalyzeTaskRequirements(task, agent)
	}

	client, err := e.llmService.GetClient(agent.LLMConfigID)
	if err != nil {
		log.Printf("获取LLM客户端失败，使用基础分析: %v", err)
		return e.AnalyzeTaskRequirements(task, agent)
	}

	prompt := e.buildLLMPrompt(task, agent)
	messages := []models.ChatMessage{
		{Role: "system", Content: "你是一个专业的任务分析助手，擅长将复杂任务拆解为可执行的子任务，并确定所需的技能类型。请以JSON格式返回分析结果。"},
		{Role: "user", Content: prompt},
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := client.Chat(ctx, messages, llm.WithTemperature(0.3), llm.WithMaxTokens(2000))
	if err != nil {
		log.Printf("LLM分析失败，使用基础分析: %v", err)
		return e.AnalyzeTaskRequirements(task, agent)
	}

	analysis, err := e.parseLLMResponse(task, response)
	if err != nil {
		log.Printf("解析LLM响应失败，使用基础分析: %v", err)
		return e.AnalyzeTaskRequirements(task, agent)
	}

	return analysis, nil
}

// buildLLMPrompt 构建LLM提示词
func (e *SkillExecutor) buildLLMPrompt(task *models.Task, agent *models.Agent) string {
	contentJSON, _ := json.MarshalIndent(task.Content, "", "  ")

	return fmt.Sprintf(`请分析以下任务并生成子任务计划：

任务信息：
- 任务ID: %s
- 任务名称: %s
- 任务描述: %s
- 任务内容: %s

可用技能类型：
- command: 系统命令执行技能
- file: 文件操作技能（读取、写入、删除、列表）
- application: 应用程序操作技能（打开浏览器、打开文件、打开应用）

请以JSON格式返回分析结果，格式如下：
{
  "required_skills": ["command", "file"],
  "subtasks": [
    {
      "name": "子任务名称",
      "description": "子任务描述",
      "skill_type": "command",
      "params": {"command": "ls -la"}
    }
  ]
}`, task.ID, task.Name, task.Description, string(contentJSON))
}

// parseLLMResponse 解析LLM响应
func (e *SkillExecutor) parseLLMResponse(task *models.Task, response string) (*TaskAnalysis, error) {
	type LLMSubtask struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		SkillType   string                 `json:"skill_type"`
		Params      map[string]interface{} `json:"params"`
	}

	type LLMResponse struct {
		RequiredSkills []string     `json:"required_skills"`
		Subtasks       []LLMSubtask `json:"subtasks"`
	}

	var llmResp LLMResponse
	err := json.Unmarshal([]byte(response), &llmResp)
	if err != nil {
		return nil, err
	}

	analysis := &TaskAnalysis{
		TaskID:          task.ID,
		RequiredSkills:  make(map[models.SkillType]bool),
		Subtasks:        make([]*Subtask, 0),
		SuggestedSkills: make([]string, 0),
	}

	for _, skillType := range llmResp.RequiredSkills {
		analysis.RequiredSkills[models.SkillType(skillType)] = true
	}

	for i, llmSubtask := range llmResp.Subtasks {
		analysis.Subtasks = append(analysis.Subtasks, &Subtask{
			ID:          fmt.Sprintf("%s-llm-%d", task.ID, i+1),
			Name:        llmSubtask.Name,
			Description: llmSubtask.Description,
			SkillType:   models.SkillType(llmSubtask.SkillType),
			Params:      llmSubtask.Params,
		})
	}

	return analysis, nil
}
