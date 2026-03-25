package intervene

import (
	"encoding/json"
	"testing"
)

func Test_keyword(t *testing.T) {
	// 示例规则：(A&B|C)&(D|E)|F
	ruleJSON := `[
			{
				"operator": "|",
				"children": [
					{
						"type": "rule",
						"rule": {
							"operator": "&",
							"children": [
								{
									"type": "rule",
									"rule": {
										"operator": "|",
										"children": [
											{
												"type": "rule",
												"rule": {
													"operator": "&",
													"children": [
														{"type": "keyword", "value": "A"},
														{"type": "keyword", "value": "B"}
													]
												}
											},
											{"type": "keyword", "value": "C"}
										]
									}
								},
								{
									"type": "rule",
									"rule": {
										"operator": "|",
										"children": [
											{"type": "keyword", "value": "D"},
											{"type": "keyword", "value": "E"}
										]
									}
								}
							]
						}
					},
					{"type": "keyword", "value": "F"}
				]
			}
		]`

	// 解析规则JSON
	var rules []KeywordRule
	if err := json.Unmarshal([]byte(ruleJSON), &rules); err != nil {
		t.Logf("解析规则失败: %v\n", err)
		return
	}

	// 测试文本 示例规则：(A&B|C)&(D|E)|F
	testTexts := []string{
		"包含A和B还有D",
		"只有C和E",
		"单独出现F",
		"A和D但没有b",
		"B、C和E",
		"什么关键词都没有",
	}

	// 执行匹配并输出结果
	for i, text := range testTexts {
		t.Logf("\n===== 测试文本 %d: %q =====\n", i+1, text)
		result := MatchRuleCheck(rules, text)
		t.Logf("测试文本 %d 最终匹配结果: %v\n", i+1, result)
	}
}

func Test_keyword_or(t *testing.T) {
	// 简化规则：A|B|C|D|E (最多5个关键词的OR规则)
	ruleJSON := `[
		{
			"operator": "|",
			"children": [
				{"type": "keyword", "value": "A"},
				{"type": "keyword", "value": "B"},
				{"type": "keyword", "value": "C"},
				{"type": "keyword", "value": "D"},
				{"type": "keyword", "value": "E"}
			]
		}
	]`

	// 解析规则JSON
	var rules []KeywordRule
	if err := json.Unmarshal([]byte(ruleJSON), &rules); err != nil {
		t.Logf("解析规则失败: %v\n", err)
		return
	}

	// 检查规则数量
	if len(rules) != 1 {
		t.Errorf("规则数量应为1，实际为: %d\n", len(rules))
		return
	}

	// 检查关键词数量
	keywordCount := 0
	for _, child := range rules[0].Children {
		if child.Type == "keyword" {
			keywordCount++
		}
	}
	if keywordCount > 5 {
		t.Errorf("关键词数量不应超过5，实际为: %d\n", keywordCount)
	}

	// 测试文本 - 简化规则：A|B|C|D|E
	testTexts := []string{
		"包含A",
		"包含B和C",
		"包含D但不包含其他",
		"包含E",
		"什么关键词都没有",
		"包含F（不在规则中）",
	}

	// 执行匹配并输出结果
	for i, text := range testTexts {
		t.Logf("\n===== 测试文本 %d: %q =====\n", i+1, text)
		result := MatchRuleCheck(rules, text)
		t.Logf("测试文本 %d 最终匹配结果: %v\n", i+1, result)
	}
}

// 额外添加一个测试函数，测试AND规则
func Test_keyword_and(t *testing.T) {
	// 简化规则：A&B&C&D&E (最多5个关键词的AND规则)
	ruleJSON := `[
		{
			"operator": "&",
			"children": [
				{"type": "keyword", "value": "A"},
				{"type": "keyword", "value": "B"},
				{"type": "keyword", "value": "C"},
				{"type": "keyword", "value": "D"},
				{"type": "keyword", "value": "E"}
			]
		}
	]`

	// 解析规则JSON
	var rules []KeywordRule
	if err := json.Unmarshal([]byte(ruleJSON), &rules); err != nil {
		t.Logf("解析规则失败: %v\n", err)
		return
	}

	// 测试文本 - AND规则：A&B&C&D&E
	testTexts := []string{
		"包含A、B、C、D和E",
		"缺少a,b,c,d,的其他关键词E",
		"只包含A",
		"什么关键词都没有",
	}

	// 执行匹配并输出结果
	for i, text := range testTexts {
		t.Logf("\n===== 测试文本 %d: %q =====\n", i+1, text)
		result := MatchRuleCheck(rules, text)
		t.Logf("测试文本 %d 最终匹配结果: %v\n", i+1, result)
	}
}

func Test_keyword_and_or(t *testing.T) {
	// 简化规则：A&B|C&D&E (两个AND组由OR连接，共5个关键词)
	ruleJSON := `[
		{
			"operator": "|",
			"children": [
				{
					"type": "rule",
					"rule": {
						"operator": "&",
						"children": [
							{"type": "keyword", "value": "A"},
							{"type": "keyword", "value": "B"}
						]
					}
				},
				{
					"type": "rule",
					"rule": {
						"operator": "&",
						"children": [
							{"type": "keyword", "value": "C"},
							{"type": "keyword", "value": "D"},
							{"type": "keyword", "value": "E"}
						]
					}
				}
			]
		}
	]`

	// 解析规则JSON
	var rules []KeywordRule
	if err := json.Unmarshal([]byte(ruleJSON), &rules); err != nil {
		t.Logf("解析规则失败: %v\n", err)
		return
	}

	// 检查规则结构
	if len(rules) != 1 {
		t.Errorf("规则数量应为1，实际为: %d\n", len(rules))
		return
	}

	// 检查关键词总数
	keywordCount := 0
	for _, child := range rules[0].Children {
		if child.Type == "rule" {
			for _, subChild := range child.Rule.Children {
				if subChild.Type == "keyword" {
					keywordCount++
				}
			}
		}
	}
	if keywordCount > 5 {
		t.Errorf("关键词数量不应超过5，实际为: %d\n", keywordCount)
	}

	// 测试文本 - 规则：A&B|C&D&E
	testTexts := []string{
		"包含A和B",
		"包含C、D和E",
		"包含A、B、C、D和E",
		"只包含A不包含b",
		"只包含C和D不包含e",
		"什么关键词都没有",
	}

	// 执行匹配并输出结果
	for i, text := range testTexts {
		t.Logf("\n===== 测试文本 %d: %q =====\n", i+1, text)
		result := MatchRuleCheck(rules, text)
		t.Logf("测试文本 %d 最终匹配结果: %v\n", i+1, result)
	}
}
