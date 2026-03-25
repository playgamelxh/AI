package database

import (
	"ai-agent-skill/internal/models"
	"database/sql"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/golang-migrate/migrate/v4"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

var (
	db    *gorm.DB
	sqlDB *sql.DB
	once  sync.Once
)

// DBConfig 数据库配置
type DBConfig struct {
	// Driver 数据库驱动
	Driver string
	// Host 数据库主机
	Host string
	// Port 数据库端口
	Port string
	// User 数据库用户名
	User string
	// Password 数据库密码
	Password string
	// DBName 数据库名称
	DBName string
	// Charset 字符集
	Charset string
	// Debug 是否启用调试模式
	Debug bool
}

// DefaultDBConfig 获取默认数据库配置
func DefaultDBConfig() *DBConfig {
	return &DBConfig{
		Driver:   "mysql",
		Host:     getEnv("DB_HOST", "localhost"),
		Port:     getEnv("DB_PORT", "3306"),
		User:     getEnv("DB_USER", "root"),
		Password: getEnv("DB_PASSWORD", ""),
		DBName:   getEnv("DB_NAME", "ai_agent"),
		Charset:  "utf8mb4",
		Debug:    getEnv("DB_DEBUG", "false") == "true",
	}
}

// getEnv 获取环境变量，默认值可选
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// DSN 生成数据库连接字符串
func (c *DBConfig) DSN() string {
	return fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?charset=%s&parseTime=True&loc=Local",
		c.User,
		c.Password,
		c.Host,
		c.Port,
		c.DBName,
		c.Charset,
	)
}

// Init 初始化数据库
func Init(cfg *DBConfig) error {
	var err error
	once.Do(func() {
		err = initDB(cfg)
	})
	return err
}

// initDB 初始化数据库连接
func initDB(cfg *DBConfig) error {
	if cfg == nil {
		cfg = DefaultDBConfig()
	}

	gormConfig := &gorm.Config{}
	if cfg.Debug {
		gormConfig.Logger = logger.Default.LogMode(logger.Info)
	} else {
		gormConfig.Logger = logger.Default.LogMode(logger.Silent)
	}

	var err error
	db, err = gorm.Open(gormMysql.Open(cfg.DSN()), gormConfig)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}

	sqlDB, err = db.DB()
	if err != nil {
		return err
	}

	sqlDB.SetMaxOpenConns(100)
	sqlDB.SetMaxIdleConns(10)

	log.Printf("Database initialized: %s@%s:%s/%s", cfg.User, cfg.Host, cfg.Port, cfg.DBName)
	return nil
}

// GetDB 获取数据库实例
func GetDB() *gorm.DB {
	return db
}

// GetSQLDB 获取原生SQL数据库实例
func GetSQLDB() *sql.DB {
	return sqlDB
}

// RunMigrations 运行数据库迁移
func RunMigrations(migrationsPath string) error {
	if sqlDB == nil {
		return fmt.Errorf("database not initialized")
	}

	driver, err := migrateMysql.WithInstance(sqlDB, &migrateMysql.Config{})
	if err != nil {
		return fmt.Errorf("failed to create migration driver: %w", err)
	}

	m, err := migrate.NewWithDatabaseInstance(
		fmt.Sprintf("file://%s", migrationsPath),
		"mysql",
		driver,
	)
	if err != nil {
		return fmt.Errorf("failed to create migration instance: %w", err)
	}

	if err := m.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	version, dirty, err := m.Version()
	if err != nil {
		log.Printf("Migration version check failed: %v", err)
	} else {
		log.Printf("Database migrated to version: %d, dirty: %v", version, dirty)
	}

	return nil
}

// AutoMigrate 使用GORM自动迁移
func AutoMigrate() error {
	if db == nil {
		return nil
	}

	log.Println("Running GORM auto migrations...")

	err := db.AutoMigrate(
		&models.LLMConfig{},
		&models.Skill{},
		&models.Agent{},
		&models.Task{},
	)

	if err != nil {
		return err
	}

	log.Println("GORM auto migrations completed successfully")
	return nil
}

// SeedData 初始化种子数据
func SeedData() error {
	if db == nil {
		return nil
	}

	log.Println("Seeding initial data...")

	if err := seedLLMConfigs(); err != nil {
		return err
	}

	if err := seedSkills(); err != nil {
		return err
	}

	if err := seedAgents(); err != nil {
		return err
	}

	log.Println("Initial data seeded successfully")
	return nil
}

// seedLLMConfigs 初始化LLM配置种子数据
func seedLLMConfigs() error {
	var count int64
	if err := db.Model(&models.LLMConfig{}).Count(&count).Error; err != nil {
		return err
	}
	if count > 0 {
		log.Println("LLM configs already exist, skipping seed")
		return nil
	}

	defaultConfigs := []*models.LLMConfig{
		{
			ID:       "ollama-default",
			Provider: models.LLMProviderOllama,
			Name:     "Ollama (本地)",
			BaseURL:  "http://localhost:11434/v1",
			Model:    "llama2",
			Enabled:  true,
		},
	}

	for _, cfg := range defaultConfigs {
		if err := db.Create(cfg).Error; err != nil {
			log.Printf("Failed to create LLM config %s: %v", cfg.Name, err)
		}
	}

	return nil
}

// seedSkills 初始化技能种子数据
func seedSkills() error {
	var count int64
	if err := db.Model(&models.Skill{}).Count(&count).Error; err != nil {
		return err
	}
	if count > 0 {
		log.Println("Skills already exist, skipping seed")
		return nil
	}

	defaultSkills := []*models.Skill{
		{
			ID:          "command",
			Name:        "系统命令执行",
			Type:        models.SkillTypeCommand,
			Description: "执行系统命令，如ls、pwd等",
			Enabled:     true,
		},
		{
			ID:          "file",
			Name:        "文件操作",
			Type:        models.SkillTypeFile,
			Description: "文件和目录操作，如创建、读取、更新、删除",
			Enabled:     true,
		},
		{
			ID:          "application",
			Name:        "应用程序操作",
			Type:        models.SkillTypeApplication,
			Description: "打开浏览器、应用程序、文件等",
			Enabled:     true,
		},
	}

	for _, skill := range defaultSkills {
		if err := db.Create(skill).Error; err != nil {
			log.Printf("Failed to create skill %s: %v", skill.Name, err)
		}
	}

	return nil
}

// seedAgents 初始化Agent种子数据
func seedAgents() error {
	var count int64
	if err := db.Model(&models.Agent{}).Count(&count).Error; err != nil {
		return err
	}
	if count > 0 {
		log.Println("Agents already exist, skipping seed")
		return nil
	}

	var llmConfig *models.LLMConfig
	if err := db.Model(&models.LLMConfig{}).First(&llmConfig).Error; err == nil {
		defaultAgent := models.NewAgent(
			"默认助手",
			"一个全能的AI助手，可以执行各种任务",
			llmConfig.ID,
		)
		defaultAgent.ID = "default-agent"

		var skills []*models.Skill
		if err := db.Model(&models.Skill{}).Find(&skills).Error; err == nil {
			for _, skill := range skills {
				defaultAgent.SkillIDs = append(defaultAgent.SkillIDs, skill.ID)
			}
		}

		if err := db.Create(defaultAgent).Error; err != nil {
			log.Printf("Failed to create default agent: %v", err)
		}
	}

	return nil
}
