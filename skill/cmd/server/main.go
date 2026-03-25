package main

import (
	"ai-agent-skill/internal/api"
	"ai-agent-skill/internal/services"
	"ai-agent-skill/pkg/config"
	"ai-agent-skill/pkg/database"
	"ai-agent-skill/pkg/scheduler"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gin-gonic/gin"
)

func main() {
	// 加载配置
	cfg := config.Load()

	// 初始化数据库
	dbCfg := database.DefaultDBConfig()
	dbCfg.Debug = cfg.Server.Mode == "debug"
	
	if err := database.Init(dbCfg); err != nil {
		log.Fatalf("数据库初始化失败: %v", err)
	}

	// 运行数据库迁移
	if err := database.RunMigrations("migrations"); err != nil {
		log.Printf("数据库迁移失败: %v", err)
		log.Println("尝试使用GORM自动迁移...")
		if err := database.AutoMigrate(); err != nil {
			log.Fatalf("GORM自动迁移也失败: %v", err)
		}
	}

	// 初始化种子数据
	if err := database.SeedData(); err != nil {
		log.Printf("种子数据初始化失败: %v", err)
	}

	// 设置Gin模式
	gin.SetMode(cfg.Server.Mode)

	// 创建Gin引擎
	r := gin.Default()

	// 创建服务
	llmService := services.NewLLMService()
	skillService := services.NewSkillService()
	agentService := services.NewAgentService()
	taskService := services.NewTaskService()

	// 创建任务调度器
	taskScheduler := scheduler.NewScheduler(taskService, agentService, skillService, llmService)
	taskScheduler.Start()
	defer taskScheduler.Stop()

	// 创建API处理器
	handler := api.NewHandler(llmService, skillService, agentService, taskService)

	// 设置路由
	api.SetupRoutes(r, handler)

	// 启动服务器
	log.Printf("AI Agent With Skill 服务启动中...")
	log.Printf("服务模式: %s", cfg.Server.Mode)
	log.Printf("监听端口: %s", cfg.Server.Port)
	log.Printf("API文档: http://localhost:%s/api/v1/health", cfg.Server.Port)

	// 优雅关闭
	go func() {
		if err := r.Run(":" + cfg.Server.Port); err != nil {
			log.Fatalf("服务启动失败: %v", err)
		}
	}()

	// 等待信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("正在关闭服务...")
}
