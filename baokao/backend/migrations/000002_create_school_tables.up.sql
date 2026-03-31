CREATE TABLE IF NOT EXISTS schools (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '学校名称',
    province_id INT NOT NULL COMMENT '所属省份ID',
    city VARCHAR(50) NOT NULL COMMENT '所在城市',
    level VARCHAR(20) COMMENT '办学层次: 本科, 专科',
    is_985 BOOLEAN DEFAULT FALSE COMMENT '是否985',
    is_211 BOOLEAN DEFAULT FALSE COMMENT '是否211',
    is_double_first_class BOOLEAN DEFAULT FALSE COMMENT '是否双一流',
    category VARCHAR(50) COMMENT '类别: 综合, 理工, 艺术, 军校等',
    batch VARCHAR(50) COMMENT '录取批次: 提前批, 艺术, 本科一批等',
    description TEXT COMMENT '学校简介',
    details LONGTEXT COMMENT '学校详情',
    website VARCHAR(100) COMMENT '学校网址',
    phone VARCHAR(50) COMMENT '招生办电话',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (province_id) REFERENCES provinces(id) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校信息表';

CREATE TABLE IF NOT EXISTS majors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    school_id INT NOT NULL COMMENT '所属学校ID',
    name VARCHAR(100) NOT NULL COMMENT '专业名称',
    description TEXT COMMENT '专业简介',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (school_id) REFERENCES schools(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校专业表';

CREATE TABLE IF NOT EXISTS major_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    school_id INT NOT NULL COMMENT '学校ID',
    major_id INT NOT NULL COMMENT '专业ID',
    province_id INT NOT NULL COMMENT '招生省份ID',
    year INT NOT NULL COMMENT '年份',
    lowest_score INT NOT NULL COMMENT '最低录取分',
    admission_count INT DEFAULT 0 COMMENT '录取人数',
    batch VARCHAR(50) COMMENT '录取批次',
    type VARCHAR(20) COMMENT '科类: 理科, 文科, 综合, 物理类, 历史类',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (school_id) REFERENCES schools(id) ON DELETE CASCADE,
    FOREIGN KEY (major_id) REFERENCES majors(id) ON DELETE CASCADE,
    FOREIGN KEY (province_id) REFERENCES provinces(id) ON DELETE CASCADE,
    UNIQUE KEY uk_school_major_province_year (school_id, major_id, province_id, year, type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学校专业各省历年录取分数表';
