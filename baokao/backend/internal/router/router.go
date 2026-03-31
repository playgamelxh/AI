package router

import (
	"baokao/internal/controller"
	"github.com/gin-gonic/gin"
)

func RegisterRoutes(r *gin.Engine) {
	api := r.Group("/api")
	
	// 省份路由
	provinces := api.Group("/provinces")
	{
		provinces.GET("", controller.GetProvinces)
		provinces.POST("", controller.CreateProvince)
		provinces.PUT("/:id", controller.UpdateProvince)
		provinces.DELETE("/:id", controller.DeleteProvince)
	}

	// 学校路由
	schools := api.Group("/schools")
	{
		schools.GET("", controller.GetSchools)
		schools.GET("/:id", controller.GetSchoolDetail)
		schools.POST("", controller.CreateSchool)
		schools.PUT("/:id", controller.UpdateSchool)
		schools.DELETE("/:id", controller.DeleteSchool)
	}

	// 专业路由
	majors := api.Group("/majors")
	{
		majors.GET("", controller.GetMajors)
		majors.POST("", controller.CreateMajor)
		majors.PUT("/:id", controller.UpdateMajor)
		majors.DELETE("/:id", controller.DeleteMajor)
	}

	// 分数路由
	scores := api.Group("/scores")
	{
		scores.GET("", controller.GetScores)
		scores.POST("", controller.CreateScore)
		scores.PUT("/:id", controller.UpdateScore)
		scores.DELETE("/:id", controller.DeleteScore)
	}

	// AI 咨询路由
	ai := api.Group("/ai")
	{
		ai.POST("/consult", controller.GetAIConsultation)
	}
}
