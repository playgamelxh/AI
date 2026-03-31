package service

import "baokao/internal/model"

func GetMajorsBySchoolID(schoolID int) ([]model.Major, error) {
	return model.GetMajorsBySchoolID(schoolID)
}

func CreateMajor(m *model.Major) error {
	return model.CreateMajor(m)
}

func UpdateMajor(m *model.Major) error {
	return model.UpdateMajor(m)
}

func DeleteMajor(id int) error {
	return model.DeleteMajor(id)
}
