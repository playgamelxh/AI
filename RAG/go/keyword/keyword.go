package main

import (
	"strings"
)

// 规则元素类型：关键词或子规则
type KeywordRuleItem struct {
	Type  string       `json:"type"`  // "keyword" 或 "rule"
	Value string       `json:"value"` // 关键词内容（当Type为keyword时）
	Rule  *KeywordRule `json:"rule"`  // 子规则（当Type为rule时）
}

// 规则结构
type KeywordRule struct {
	Operator string            `json:"operator"` // "&" 或 "|"
	Children []KeywordRuleItem `json:"children"` // 子元素（关键词或子规则）
}

// 直接判断文本是否包含关键词（区分大小写）
func containsKeyword(text, keyword string) bool {
	return strings.Contains(text, keyword)
}

// 递归匹配单个规则（增加调试日志）
func matchRule(rule KeywordRule, text string, depth int) bool {
	if len(rule.Children) == 0 {
		return true
	}

	// 收集所有子元素的匹配结果
	childResults := make([]bool, 0, len(rule.Children))
	for _, child := range rule.Children {
		var res bool
		switch child.Type {
		case "keyword":
			res = containsKeyword(text, child.Value)
		case "rule":
			if child.Rule == nil {
				res = false
			} else {
				res = matchRule(*child.Rule, text, depth+1)
			}
		default:
			res = false
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

	return result
}

// 匹配规则集：任意一个规则匹配即返回true
func MatchRuleCheck(rules []KeywordRule, text string) bool {
	// 规则集为空时不匹配
	if len(rules) == 0 {
		return true
	}

	// 匹配每个规则，任意一个匹配则返回true
	for _, rule := range rules {
		if matchRule(rule, text, 1) {
			return true
		}
	}
	return false
}
