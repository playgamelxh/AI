package service

import "baokao/internal/model"

func GetSchools(city string, category string) ([]model.School, error) {
	// 这里可以加入更复杂的业务逻辑，如缓存等
	return model.GetSchools(city, category)
}

func GetSchoolDetail(id int) (*model.School, error) {
	return model.GetSchoolByID(id)
}

func CreateSchool(s *model.School) error {
	return model.CreateSchool(s)
}

func UpdateSchool(s *model.School) error {
	return model.UpdateSchool(s)
}

func DeleteSchool(id int) error {
	return model.DeleteSchool(id)
}
