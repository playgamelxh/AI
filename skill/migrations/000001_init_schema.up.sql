-- 创建 llm_configs 表
CREATE TABLE IF NOT EXISTS llm_configs (
    id VARCHAR(255) PRIMARY KEY COMMENT '配置唯一标识',
    provider VARCHAR(50) NOT NULL COMMENT '模型提供商类型',
    name VARCHAR(255) NOT NULL COMMENT '配置名称',
    api_key TEXT COMMENT 'API密钥',
    base_url VARCHAR(500) COMMENT 'API基础地址',
    model VARCHAR(255) NOT NULL COMMENT '模型名称',
    temperature DECIMAL(3,2) DEFAULT 0.70 COMMENT '温度参数',
    max_tokens INT DEFAULT 2048 COMMENT '最大token数',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    deleted_at DATETIME(3) DEFAULT NULL COMMENT '删除时间',
    INDEX idx_provider (provider),
    INDEX idx_deleted_at (deleted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='大模型配置表';

-- 创建 skills 表
CREATE TABLE IF NOT EXISTS skills (
    id VARCHAR(255) PRIMARY KEY COMMENT '技能唯一标识',
    name VARCHAR(255) NOT NULL COMMENT '技能名称',
    type VARCHAR(50) NOT NULL COMMENT '技能类型',
    description TEXT COMMENT '技能描述',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    config_json TEXT COMMENT '技能配置JSON',
    created_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    updated_at DATETIME(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
    deleted_at DATETIME(3) DEFAULT NULL COMMENT '删除时间',
    INDEX idx_type (type),
    INDEX idx_deleted_at (deleted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='技能表';
