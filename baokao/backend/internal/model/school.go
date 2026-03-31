package model

import (
	"baokao/internal/db"
)

type School struct {
	ID                 int    `json:"id" db:"id"`
	Name               string `json:"name" db:"name" binding:"required"`
	ProvinceID         int    `json:"province_id" db:"province_id" binding:"required"`
	City               string `json:"city" db:"city" binding:"required"`
	Level              string `json:"level" db:"level"` // 本科, 专科
	Is985              bool   `json:"is_985" db:"is_985"`
	Is211              bool   `json:"is_211" db:"is_211"`
	IsDoubleFirstClass bool   `json:"is_double_first_class" db:"is_double_first_class"`
	Category           string `json:"category" db:"category"` // 综合, 理工, 艺术, 军校等
	Batch              string `json:"batch" db:"batch"`       // 提前批, 艺术, 本科一批等
	Description        string `json:"description" db:"description"`
	Details            string `json:"details" db:"details"`
	Website            string `json:"website" db:"website"`
	Phone              string `json:"phone" db:"phone"`
}

func GetSchools(city string, category string) ([]School, error) {
	query := "SELECT id, name, province_id, city, level, is_985, is_211, is_double_first_class, category, batch, description, details, website, phone FROM schools WHERE 1=1"
	var args []interface{}

	if city != "" {
		query += " AND city LIKE ?"
		args = append(args, "%"+city+"%")
	}
	if category != "" {
		query += " AND category = ?"
		args = append(args, category)
	}
	
	query += " ORDER BY id DESC"

	rows, err := db.DB.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []School
	for rows.Next() {
		var s School
		err := rows.Scan(&s.ID, &s.Name, &s.ProvinceID, &s.City, &s.Level, &s.Is985, &s.Is211, &s.IsDoubleFirstClass, &s.Category, &s.Batch, &s.Description, &s.Details, &s.Website, &s.Phone)
		if err != nil {
			continue
		}
		list = append(list, s)
	}
	if list == nil {
		list = []School{}
	}
	return list, nil
}

func GetSchoolByID(id int) (*School, error) {
	var s School
	err := db.DB.QueryRow("SELECT id, name, province_id, city, level, is_985, is_211, is_double_first_class, category, batch, description, details, website, phone FROM schools WHERE id = ?", id).
		Scan(&s.ID, &s.Name, &s.ProvinceID, &s.City, &s.Level, &s.Is985, &s.Is211, &s.IsDoubleFirstClass, &s.Category, &s.Batch, &s.Description, &s.Details, &s.Website, &s.Phone)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

func CreateSchool(s *School) error {
	query := `INSERT INTO schools (name, province_id, city, level, is_985, is_211, is_double_first_class, category, batch, description, details, website, phone) 
	          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
	result, err := db.DB.Exec(query, s.Name, s.ProvinceID, s.City, s.Level, s.Is985, s.Is211, s.IsDoubleFirstClass, s.Category, s.Batch, s.Description, s.Details, s.Website, s.Phone)
	if err != nil {
		return err
	}
	id, _ := result.LastInsertId()
	s.ID = int(id)
	return nil
}

func UpdateSchool(s *School) error {
	query := `UPDATE schools SET name=?, province_id=?, city=?, level=?, is_985=?, is_211=?, is_double_first_class=?, category=?, batch=?, description=?, details=?, website=?, phone=? WHERE id=?`
	_, err := db.DB.Exec(query, s.Name, s.ProvinceID, s.City, s.Level, s.Is985, s.Is211, s.IsDoubleFirstClass, s.Category, s.Batch, s.Description, s.Details, s.Website, s.Phone, s.ID)
	return err
}

func DeleteSchool(id int) error {
	_, err := db.DB.Exec("DELETE FROM schools WHERE id = ?", id)
	return err
}
