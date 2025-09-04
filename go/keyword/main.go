package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// 规则元素类型：关键词或子规则
type RuleElement struct {
	Type  string `json:"type"`  // "keyword" 或 "rule"
	Value string `json:"value"` // 关键词内容（当Type为keyword时）
	Rule  *Rule  `json:"rule"`  // 子规则（当Type为rule时）
}

// 规则结构
type Rule struct {
	Operator string        `json:"operator"` // "&" 或 "|"
	Children []RuleElement `json:"children"` // 子元素（关键词或子规则）
}

// 直接判断文本是否包含关键词（区分大小写）
func containsKeyword(text, keyword string) bool {
	return strings.Contains(text, keyword)
}

// 递归匹配单个规则（增加调试日志）
func matchRule(rule Rule, text string, depth int) bool {
	indent := strings.Repeat("  ", depth) // 缩进显示，便于观察嵌套层级
	if len(rule.Children) == 0 {
		fmt.Printf("%s规则匹配: 空规则 → 返回 false\n", indent)
		return false
	}
	
	// 收集所有子元素的匹配结果
	childResults := make([]bool, 0, len(rule.Children))
	for i, child := range rule.Children {
		var res bool
		switch child.Type {
		case "keyword":
			res = containsKeyword(text, child.Value)
			fmt.Printf("%s子元素 %d (关键词 %q): %v\n", indent, i, child.Value, res)
		case "rule":
			if child.Rule == nil {
				res = false
				fmt.Printf("%s子元素 %d (空规则): %v\n", indent, i, res)
			} else {
				fmt.Printf("%s子元素 %d (子规则, operator=%s):\n", indent, i, child.Rule.Operator)
				res = matchRule(*child.Rule, text, depth+1)
				fmt.Printf("%s子元素 %d (子规则) 结果: %v\n", indent, i, res)
			}
		default:
			res = false
			fmt.Printf("%s子元素 %d (未知类型): %v\n", indent, i, res)
		}
		childResults = append(childResults, res)
	}
	
	// 根据运算符计算最终结果
	var result bool
	switch rule.Operator {
	case "&":
		result = true
		for _, res := range childResults {
			if !res {
				result = false
				break
			}
		}
	case "|":
		result = false
		for _, res := range childResults {
			if res {
				result = true
				break
			}
		}
	default:
		result = false
	}
	
	fmt.Printf("%s当前规则 (operator=%s) 结果: %v\n", indent, rule.Operator, result)
	return result
}

// 匹配规则集：任意一个规则匹配即返回true
func MatchRuleSet(rules []Rule, text string) bool {
	// 规则集为空时不匹配
	
	// 匹配每个规则，任意一个匹配则返回true
	for i, rule := range rules {
		fmt.Printf("=== 匹配规则集第 %d 个规则 ===\n", i+1)
		if matchRule(rule, text, 1) {
			return true
		}
	}
	return false
}

func main() {
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
	var rules []Rule
	if err := json.Unmarshal([]byte(ruleJSON), &rules); err != nil {
		fmt.Printf("解析规则失败: %v\n", err)
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
		fmt.Printf("\n===== 测试文本 %d: %q =====\n", i+1, text)
		result := MatchRuleSet(rules, text)
		fmt.Printf("测试文本 %d 最终匹配结果: %v\n", i+1, result)
	}
}
