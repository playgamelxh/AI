package config

import (
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// ServerConfig 服务器配置
type ServerConfig struct {
	// Port 服务端口
	Port string `yaml:"port"`
	// Mode 运行模式 (debug/release)
	Mode string `yaml:"mode"`
}

// AppConfig 应用配置
type AppConfig struct {
	// Server 服务器配置
	Server ServerConfig `yaml:"server"`
	// ConfigPath 配置文件路径
	ConfigPath string `yaml:"-"`
}

var (
	instance *AppConfig
	once     sync.Once
)

// Load 加载配置文件
func Load(configPath ...string) *AppConfig {
	once.Do(func() {
		path := "config.yaml"
		if len(configPath) > 0 && configPath[0] != "" {
			path = configPath[0]
		}

		instance = &AppConfig{
			ConfigPath: path,
			Server: ServerConfig{
				Port: "8080",
				Mode: "debug",
			},
		}

		// 尝试从文件加载配置
		if _, err := os.Stat(path); err == nil {
			data, err := os.ReadFile(path)
			if err == nil {
				_ = yaml.Unmarshal(data, instance)
			}
		}

		// 从环境变量覆盖配置
		if port := os.Getenv("PORT"); port != "" {
			instance.Server.Port = port
		}
		if mode := os.Getenv("GIN_MODE"); mode != "" {
			instance.Server.Mode = mode
		}
	})
	return instance
}

// Get 获取全局配置实例
func Get() *AppConfig {
	return instance
}

// Save 保存配置到文件
func (c *AppConfig) Save() error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(c.ConfigPath, data, 0644)
}
