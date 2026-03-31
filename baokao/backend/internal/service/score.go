package service

import "baokao/internal/model"

func GetMajorScores(schoolID int, provinceID int, year int) ([]model.MajorScore, error) {
	return model.GetMajorScores(schoolID, provinceID, year)
}

func CreateMajorScore(s *model.MajorScore) error {
	return model.CreateMajorScore(s)
}

func UpdateMajorScore(s *model.MajorScore) error {
	return model.UpdateMajorScore(s)
}

func DeleteMajorScore(id int) error {
	return model.DeleteMajorScore(id)
}
