package controller

import (
	"baokao/internal/model"
	"baokao/internal/service"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

func GetScores(c *gin.Context) {
	schoolIDStr := c.Query("school_id")
	provinceIDStr := c.Query("province_id")
	yearStr := c.Query("year")

	schoolID, _ := strconv.Atoi(schoolIDStr)
	provinceID, _ := strconv.Atoi(provinceIDStr)
	year, _ := strconv.Atoi(yearStr)

	list, err := service.GetMajorScores(schoolID, provinceID, year)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"data": list})
}

func CreateScore(c *gin.Context) {
	var req model.MajorScore
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := service.CreateMajorScore(&req); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"data": req})
}

func UpdateScore(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID"})
		return
	}

	var req model.MajorScore
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req.ID = id

	if err := service.UpdateMajorScore(&req); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"data": "success"})
}

func DeleteScore(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID"})
		return
	}

	if err := service.DeleteMajorScore(id); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"data": "success"})
}
