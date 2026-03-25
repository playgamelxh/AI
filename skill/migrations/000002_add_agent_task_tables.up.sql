-- 创建 agents 表
CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(255) PRIMARY KEY COMMENT '智能体唯一标识',
    name VARCHAR(255) NOT NULL COMMENT '智能体名称',
    description TEXT COMMENT '智能体描述',
    llm_config_id VARCHAR(255) COMMENT '绑定的大模型配置ID',
    skill_ids_json TEXT COMMENT '绑定的技能ID列表JSON',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    deleted_at DATETIME(3) DEFAULT NULL COMMENT '删除时间',
    INDEX idx_llm_config_id (llm_config_id),
    INDEX idx_deleted_at (deleted_at),
    FOREIGN KEY (llm_config_id) REFERENCES llm_configs(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI智能体表';

-- 创建 tasks 表
CREATE TABLE IF NOT EXISTS tasks (
    id VARCHAR(255) PRIMARY KEY COMMENT '任务唯一标识',
    agent_id VARCHAR(255) NOT NULL COMMENT '关联的智能体ID',
    name VARCHAR(255) NOT NULL COMMENT '任务名称',
    description TEXT COMMENT '任务描述',
    type VARCHAR(50) NOT NULL COMMENT '任务类型',
    status VARCHAR(50) NOT NULL COMMENT '任务状态',
    cron_expr VARCHAR(100) COMMENT 'Cron表达式',
    scheduled_time DATETIME(3) COMMENT '计划执行时间',
    last_run_time DATETIME(3) COMMENT '最后一次执行时间',
    next_run_time DATETIME(3) COMMENT '下一次执行时间',
    content_json TEXT COMMENT '任务内容JSON',
    result_json TEXT COMMENT '任务结果JSON',
    error_message TEXT COMMENT '错误信息',
    retry_count INT DEFAULT 0 COMMENT '重试次数',
    max_retries INT DEFAULT 3 COMMENT '最大重试次数',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    deleted_at DATETIME(3) DEFAULT NULL COMMENT '删除时间',
    INDEX idx_agent_id (agent_id),
    INDEX idx_type (type),
    INDEX idx_status (status),
    INDEX idx_scheduled_time (scheduled_time),
    INDEX idx_next_run_time (next_run_time),
    INDEX idx_deleted_at (deleted_at),
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务表';
