package controller

import (
	"baokao/internal/service"
	"net/http"

	"github.com/gin-gonic/gin"
)

type AIConsultReq struct {
	Score     int    `json:"score" binding:"required"`
	Province  string `json:"province" binding:"required"`
	Type      string `json:"type" binding:"required"` // 理科/文科/物理类/历史类
	Interests string `json:"interests"`
	CityPref  string `json:"city_pref"`
	MajorPref string `json:"major_pref"`
	Subjects  string `json:"subjects"`
}

func GetAIConsultation(c *gin.Context) {
	var req AIConsultReq
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 调用 Service 层处理 AI 推荐逻辑
	result, err := service.GenerateAIPlan(req.Score, req.Province, req.Type, req.Interests, req.CityPref, req.MajorPref, req.Subjects)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "AI大模型调用失败: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"data": result})
}
