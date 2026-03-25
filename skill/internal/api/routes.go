package api

import (
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// SetupRoutes 设置API路由
func SetupRoutes(r *gin.Engine, handler *Handler) {
	// CORS配置
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
	}))

	// API路由组
	api := r.Group("/api/v1")
	{
		// 健康检查
		api.GET("/health", handler.Health)

		// LLM配置管理
		llm := api.Group("/llm")
		{
			llm.GET("/configs", handler.ListLLMConfigs)
			llm.GET("/configs/:id", handler.GetLLMConfig)
			llm.POST("/configs", handler.CreateLLMConfig)
			llm.PUT("/configs/:id", handler.UpdateLLMConfig)
			llm.DELETE("/configs/:id", handler.DeleteLLMConfig)
			llm.POST("/configs/:id/toggle", handler.ToggleLLMConfig)
			llm.POST("/chat", handler.Chat)
		}

		// 技能管理
		skills := api.Group("/skills")
		{
			skills.GET("", handler.ListSkills)
			skills.GET("/:id", handler.GetSkill)
			skills.POST("", handler.CreateSkill)
			skills.PUT("/:id", handler.UpdateSkill)
			skills.DELETE("/:id", handler.DeleteSkill)
			skills.POST("/:id/toggle", handler.ToggleSkill)
		}

		// Agent管理
		agents := api.Group("/agents")
		{
			agents.GET("", handler.ListAgents)
			agents.GET("/:id", handler.GetAgent)
			agents.POST("", handler.CreateAgent)
			agents.PUT("/:id", handler.UpdateAgent)
			agents.DELETE("/:id", handler.DeleteAgent)
			agents.POST("/:id/toggle", handler.ToggleAgent)
			agents.POST("/:id/skills", handler.AddSkillToAgent)
			agents.DELETE("/:id/skills", handler.RemoveSkillFromAgent)
		}

		// 任务管理
		tasks := api.Group("/tasks")
		{
			tasks.GET("", handler.ListTasks)
			tasks.GET("/agent/:agent_id", handler.GetTasksByAgent)
			tasks.GET("/:id", handler.GetTask)
			tasks.POST("", handler.CreateTask)
			tasks.PUT("/:id", handler.UpdateTask)
			tasks.DELETE("/:id", handler.DeleteTask)
			tasks.POST("/:id/toggle", handler.ToggleTask)
		}
	}
}
