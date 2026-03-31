package service

import (
	"fmt"
)

type AIPlanResult struct {
	Summary     string   `json:"summary"`
	BatchA      []string `json:"batch_a"`
	Parallel    []string `json:"parallel"`
	Suggestions string   `json:"suggestions"`
}

// GenerateAIPlan 调用AI大模型生成报考计划 (目前返回 Mock 数据)
func GenerateAIPlan(score int, province, typeClass, interests, cityPref, majorPref, subjects string) (*AIPlanResult, error) {
	// 实际开发中这里将调用外部大模型 API (如文心一言, ChatGPT等)
	// 这里目前使用简单的 Mock 逻辑返回
	
	summary := fmt.Sprintf("根据您的分数(%d分)和偏好(城市:%s, 专业:%s)，我们为您量身定制了以下报考计划：", score, cityPref, majorPref)
	
	var batchA, parallel []string
	
	if score > 650 {
		batchA = []string{"清华大学 - 计算机科学与技术", "北京大学 - 汉语言文学", "复旦大学 - 新闻学"}
		parallel = []string{"上海交通大学 - 机械工程", "浙江大学 - 临床医学"}
	} else if score > 550 {
		batchA = []string{"北京邮电大学 - 软件工程", "武汉大学 - 通信工程"}
		parallel = []string{"电子科技大学 - 计算机类", "西安电子科技大学 - 电子信息"}
	} else {
		batchA = []string{"深圳职业技术大学 - 软件技术", "广东轻工职业技术学院 - 计算机网络"}
		parallel = []string{"南京工业职业技术大学 - 大数据技术", "北京电子科技职业学院 - 物联网应用"}
	}

	suggestions := "建议冲刺第一批次的学校，同时用平行志愿保底，确保被录取。同时注意您的兴趣爱好与专业方向的匹配度。"

	return &AIPlanResult{
		Summary:     summary,
		BatchA:      batchA,
		Parallel:    parallel,
		Suggestions: suggestions,
	}, nil
}
