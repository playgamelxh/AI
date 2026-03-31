package service

import "baokao/internal/model"

func GetProvinces() ([]model.Province, error) {
	return model.GetAllProvinces()
}

func CreateProvince(p *model.Province) error {
	return model.CreateProvince(p)
}

func UpdateProvince(p *model.Province) error {
	return model.UpdateProvince(p)
}

func DeleteProvince(id int) error {
	return model.DeleteProvince(id)
}
