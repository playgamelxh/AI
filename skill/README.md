## AI Agent With Skill项目

### Description
* 这是一个AI Agent项目，使用Golang语言实现，
* 代用大模型兼容市面常见的大模型，比如ChatGPT、Ollama、DeepSeek、Doubao、Qwen、Gemini、Claude、LLaMA、Grok、元宝、GLM、Kimi等
* Golang实现，使用gin框架实现HTTP API，拥有管理页面，用于配置不同大模型的参数和skill技能
* 前端框架使用Vue3实现，页面风格清新淡雅，操作简单，不需要登录认证，直接访问即可
* 拥有skill模块，用于管理不同的skill技能和加载不同的skill技能，不同的大模型配置等
* 使用MySQL数据库存储配置和技能数据，使用golang-migrate进行数据库版本管理
* 使用docker-compose部署，包括后端服务和前端服务
* 项目结构清晰，代码质量高，注释详细，方便维护和扩展
* 实现数据库迁移管理系统，使用golang-migrate
* 创建数据库迁移目录 (migrations/)
* 创建初始迁移文件：
  - 000001_init_schema.up.sql - 创建llm_configs和skills表
  - 000001_init_schema.down.sql - 回滚初始表结构
* 更新数据模型以适配MySQL：
  - 使用DATETIME(3)时间戳格式
  - 使用utf8mb4字符集
  - 添加适当的索引
  - 确保数据库表结构与模型保持一致
* 实现agent管理模块，可以增加、删除agent，修改agent的大模型配置和skill技能
* agent增加任务模块，给agent添加任务，定时检测任务状态，定期检查任务是否完成，如果需要执行任务内容，检索任务内容并执行skill技能，完成任务后，检查任务完成情况，更新任务状态，任务可以分为一次性任务，或者周期任务。

### 项目结构
```
ai-agent-skill/
├── cmd/
│   └── server/
│       └── main.go              # 后端服务入口
├── internal/
│   ├── api/
│   │   ├── handlers.go          # API处理器
│   │   └── routes.go            # 路由配置
│   ├── models/
│   │   └── config.go            # 数据模型定义
│   └── services/
│       ├── llm_service.go       # LLM配置管理服务
│       └── skill_service.go     # 技能管理服务
├── pkg/
│   ├── config/
│   │   └── config.go            # 配置管理
│   ├── database/
│   │   └── database.go          # 数据库管理（迁移、种子数据）
│   ├── scheduler/
│   │   └── scheduler.go         # 任务调度器
│   └── llm/
│       ├── interface.go         # LLM客户端接口定义
│       ├── factory.go           # LLM客户端工厂
│       └── openai_client.go     # OpenAI兼容客户端实现
├── migrations/                  # 数据库迁移文件
│   ├── 000001_init_schema.up.sql
│   └── 000001_init_schema.down.sql
├── web/
│   ├── src/
│   │   ├── App.vue              # 前端主组件
│   │   └── main.js              # 前端入口
│   ├── index.html               # HTML入口
│   ├── package.json             # 前端依赖配置
│   └── vite.config.js           # Vite配置
├── Dockerfile.backend           # 后端Docker镜像
├── Dockerfile.frontend          # 前端Docker镜像
├── docker-compose.yml           # Docker Compose配置
├── nginx.conf                   # Nginx配置
├── .env.example                 # 环境变量配置示例
├── go.mod                       # Go模块依赖
├── .gitignore                   # Git忽略文件
└── README.md                    # 项目文档
```

### 快速开始

#### 前置要求
- Go 1.21+
- Node.js 18+
- MySQL 5.7+ 或 8.0+
- Docker (可选，用于Docker Compose部署)

#### 数据库准备
```bash
# 创建数据库
mysql -u root -p
CREATE DATABASE ai_agent DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

# 或者使用环境变量配置
cp .env.example .env
# 编辑.env文件配置数据库连接信息
```

#### 后端开发
```bash
# 直接运行
go run cmd/server/main.go

# 或构建后运行
go build -o ai-agent-skill cmd/server/main.go
./ai-agent-skill
```

#### 前端开发
```bash
cd web
npm install
npm run dev
```

#### Docker Compose部署
```bash
docker-compose up -d
```

#### 访问地址
- 后端API: http://localhost:8080/api/v1
- 前端页面: http://localhost:5173
- 健康检查: http://localhost:8080/api/v1/health

### API文档

#### 健康检查
- `GET /api/v1/health` - 服务健康检查

#### 大模型配置管理
- `GET /api/v1/llm/configs` - 获取所有LLM配置
- `GET /api/v1/llm/configs/:id` - 获取单个LLM配置
- `POST /api/v1/llm/configs` - 创建LLM配置
- `PUT /api/v1/llm/configs/:id` - 更新LLM配置
- `DELETE /api/v1/llm/configs/:id` - 删除LLM配置
- `POST /api/v1/llm/configs/:id/toggle` - 切换LLM配置启用状态
- `POST /api/v1/llm/chat` - 与LLM对话

#### 技能管理
- `GET /api/v1/skills` - 获取所有技能
- `GET /api/v1/skills/:id` - 获取单个技能
- `POST /api/v1/skills` - 创建技能
- `PUT /api/v1/skills/:id` - 更新技能
- `DELETE /api/v1/skills/:id` - 删除技能
- `POST /api/v1/skills/:id/toggle` - 切换技能启用状态

#### Agent管理
- `GET /api/v1/agents` - 获取所有Agent
- `GET /api/v1/agents/:id` - 获取单个Agent
- `POST /api/v1/agents` - 创建Agent
- `PUT /api/v1/agents/:id` - 更新Agent
- `DELETE /api/v1/agents/:id` - 删除Agent
- `POST /api/v1/agents/:id/toggle` - 切换Agent启用状态
- `POST /api/v1/agents/:id/skills` - 给Agent添加技能
- `DELETE /api/v1/agents/:id/skills` - 从Agent移除技能

#### 任务管理
- `GET /api/v1/tasks` - 获取所有任务
- `GET /api/v1/tasks/agent/:agent_id` - 获取Agent的任务
- `GET /api/v1/tasks/:id` - 获取单个任务
- `POST /api/v1/tasks` - 创建任务
- `PUT /api/v1/tasks/:id` - 更新任务
- `DELETE /api/v1/tasks/:id` - 删除任务
- `POST /api/v1/tasks/:id/toggle` - 切换任务启用状态

### 支持的大模型提供商
- OpenAI / ChatGPT
- Ollama (本地模型)
- DeepSeek
- 豆包 (火山引擎)
- 通义千问
- Google DeepMind（Gemini）
- Anthropic（Claude）
- Meta（LLaMA）
- XAI（Grok）
- 腾讯（元宝）
-  智谱 AI（GLM）
- 月之暗面（Kimi）
---

## 迭代日志

#### v1.0.0 - 2026-03-19
- 初始化项目基础结构
- 创建Go模块依赖配置 (go.mod)
- 创建Git忽略文件 (.gitignore)
- 实现数据模型层 (internal/models/config.go)
  - 定义LLM配置模型，支持5种提供商
  - 定义技能模型，支持3种技能类型
  - 定义聊天消息和API响应模型
  - 完整的中文注释
- 实现配置管理 (pkg/config/config.go)
  - 支持YAML配置文件
  - 支持环境变量覆盖
  - 单例模式实现
- 实现LLM客户端层 (pkg/llm/)
  - 统一的LLM客户端接口定义
  - OpenAI兼容客户端实现，支持所有兼容OpenAI API的提供商
  - 客户端工厂，根据配置自动创建对应客户端
  - 流式输出支持
  - 完整的中文注释
- 实现服务层 (internal/services/)
  - LLM配置管理服务，支持CRUD和状态切换
  - 技能管理服务，支持CRUD和状态切换
  - 默认配置和技能初始化
  - 线程安全的并发控制
- 实现API层 (internal/api/)
  - 完整的API处理器
  - RESTful路由配置
  - CORS跨域支持
  - 统一的响应格式
- 实现后端服务入口 (cmd/server/main.go)
  - Gin框架HTTP服务器
  - 服务启动日志
- 初始化前端项目 (web/)
  - Vue3 + Vite 项目结构
  - Element Plus UI组件库
  - 清新淡雅的渐变背景风格
  - 基础标签页布局
- 创建Docker配置文件
  - Dockerfile.backend - 后端镜像
  - Dockerfile.frontend - 前端镜像
  - docker-compose.yml - 编排配置
  - nginx.conf - Nginx反向代理配置
- 更新项目文档
  - 完整的项目结构说明
  - 快速开始指南
  - API文档
  - v1.0.0迭代日志

#### v2.0.0 - 2026-03-19
- 更新项目结构说明，添加数据库模块
- 扩展支持的大模型提供商，从5个增加到12个：
  - 新增：Google DeepMind（Gemini）
  - 新增：Anthropic（Claude）
  - 新增：Meta（LLaMA）
  - 新增：XAI（Grok）
  - 新增：腾讯（元宝）
  - 新增：智谱 AI（GLM）
  - 新增：月之暗面（Kimi）
- 实现数据库支持，使用SQLite作为存储引擎
- 创建数据库管理模块 (pkg/database/database.go)
  - GORM ORM框架集成
  - 数据库连接池管理
  - 自动迁移（AutoMigrate）
  - 种子数据初始化（SeedData）
  - 单例模式数据库连接
- 更新数据模型添加GORM支持
  - LLMConfig模型添加数据库标签
  - Skill模型添加数据库标签和ConfigJSON字段
  - 添加CreatedAt、UpdatedAt、DeletedAt时间戳
  - 支持软删除功能
- 更新LLM客户端工厂，添加新提供商支持
  - Gemini客户端创建方法
  - Claude客户端创建方法
  - LLaMA客户端创建方法
  - Grok客户端创建方法
  - 元宝客户端创建方法
  - GLM客户端创建方法
  - Kimi客户端创建方法
- 更新服务层使用数据库持久化
  - LLMService改为从数据库读写
  - SkillService改为从数据库读写
  - Skill配置JSON序列化/反序列化
  - 完整的错误处理
- 更新API处理器处理数据库错误
  - 所有接口添加错误返回处理
  - 统一的错误响应格式
- 更新主程序 (cmd/server/main.go)
  - 数据库初始化
  - 自动迁移执行
  - 种子数据初始化
- 更新Go模块依赖
  - 添加gorm.io/gorm
  - 添加gorm.io/driver/sqlite
- 更新.gitignore添加data目录
- 完整的中文代码注释

#### v3.0.0 - 2026-03-19
- 将数据库从SQLite迁移到MySQL
- 实现数据库迁移管理系统，使用golang-migrate
- 创建数据库迁移目录 (migrations/)
- 创建初始迁移文件：
  - 000001_init_schema.up.sql - 创建llm_configs和skills表
  - 000001_init_schema.down.sql - 回滚初始表结构
- 更新数据模型以适配MySQL：
  - 使用DATETIME(3)时间戳格式
  - 使用utf8mb4字符集
  - 添加适当的索引
- 重构数据库管理模块 (pkg/database/database.go)
  - 支持MySQL连接配置
  - 支持环境变量配置（DB_HOST、DB_PORT、DB_USER、DB_PASSWORD、DB_NAME、DB_DEBUG）
  - 默认配置提供器
  - DSN字符串生成方法
  - RunMigrations() 方法运行golang-migrate迁移
  - 保留AutoMigrate()作为降级方案
  - 修复mysql包导入冲突问题
- 更新Go模块依赖：
  - 移除 gorm.io/driver/sqlite
  - 添加 gorm.io/driver/mysql v1.5.2
  - 添加 github.com/golang-migrate/migrate/v4 v4.17.0
- 更新主程序 (cmd/server/main.go)
  - 使用DefaultDBConfig()获取配置
  - 优先使用golang-migrate迁移
  - 失败时降级使用GORM AutoMigrate
- 创建环境变量配置示例文件 (.env.example)
- 更新.gitignore：
  - 移除data/、*.db等SQLite相关规则
  - 添加.env并保留.env.example
- 更新README.md项目结构说明，添加migrations目录
- 完整的MySQL数据库表结构设计
- 支持数据库版本追踪和回滚
- 完整的中文代码注释

#### v4.0.0 - 2026-03-19
- 实现Agent管理模块
  - 创建Agent数据模型 (internal/models/config.go)
  - Agent支持绑定LLM配置
  - Agent支持绑定多个Skill技能
  - Agent CRUD操作（增删改查）
  - Agent启用/禁用切换
  - Agent技能添加/移除
- 实现Task任务模块
  - 创建Task数据模型 (internal/models/config.go)
  - 支持一次性任务 (once) 和周期性任务 (periodic)
  - 任务状态管理（待执行、执行中、已完成、失败、已暂停）
  - 支持Cron表达式（周期性任务）
  - 支持计划执行时间
  - 支持任务内容和结果JSON存储
  - 支持任务重试机制（配置最大重试次数）
  - Task CRUD操作（增删改查）
  - Task启用/禁用切换
  - 根据Agent查询任务
- 创建Agent服务 (internal/services/agent_service.go)
  - 完整的Agent管理逻辑
  - JSON字段序列化/反序列化
  - 线程安全的并发控制
- 创建Task服务 (internal/services/task_service.go)
  - 完整的Task管理逻辑
  - 获取待执行任务
  - 更新任务状态
  - JSON字段序列化/反序列化
  - 线程安全的并发控制
- 创建任务调度器 (pkg/scheduler/scheduler.go)
  - 定时检查待执行任务
  - 自动执行到期任务
  - 任务执行逻辑框架
  - 支持任务失败重试
  - 周期性任务自动调度下次运行
  - 优雅的启动和停止机制
- 更新API处理器 (internal/api/handlers.go)
  - 添加Agent管理API
  - 添加Task管理API
  - 完整的错误处理
- 更新API路由 (internal/api/routes.go)
  - Agent管理路由组 (/api/v1/agents)
  - Task管理路由组 (/api/v1/tasks)
- 创建数据库迁移文件
  - 000002_add_agent_task_tables.up.sql - 创建agents和tasks表
  - 000002_add_agent_task_tables.down.sql - 回滚Agent和Task表
- 更新数据库管理模块 (pkg/database/database.go)
  - AutoMigrate添加Agent和Task模型
  - SeedData添加Agent种子数据
  - 创建默认Agent，绑定所有技能
- 更新主程序 (cmd/server/main.go)
  - 初始化Agent和Task服务
  - 创建并启动任务调度器
  - 优雅关闭处理（监听SIGINT/SIGTERM）
  - 更新Handler初始化，传入新服务
- 更新项目结构 (README.md)
  - 添加scheduler和migrations目录
  - 更新API文档，添加Agent和Task接口
- 完整的中文代码注释