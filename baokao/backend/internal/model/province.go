package model

import "baokao/internal/db"

type Province struct {
	ID   int    `json:"id" db:"id"`
	Code string `json:"code" db:"code" binding:"required"`
	Name string `json:"name" db:"name" binding:"required"`
}

func GetAllProvinces() ([]Province, error) {
	rows, err := db.DB.Query("SELECT id, code, name FROM provinces ORDER BY id ASC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []Province
	for rows.Next() {
		var p Province
		if err := rows.Scan(&p.ID, &p.Code, &p.Name); err != nil {
			continue
		}
		list = append(list, p)
	}

	if list == nil {
		list = []Province{}
	}
	return list, nil
}

func CreateProvince(p *Province) error {
	result, err := db.DB.Exec("INSERT INTO provinces (code, name) VALUES (?, ?)", p.Code, p.Name)
	if err != nil {
		return err
	}
	id, _ := result.LastInsertId()
	p.ID = int(id)
	return nil
}

func UpdateProvince(p *Province) error {
	_, err := db.DB.Exec("UPDATE provinces SET code = ?, name = ? WHERE id = ?", p.Code, p.Name, p.ID)
	return err
}

func DeleteProvince(id int) error {
	_, err := db.DB.Exec("DELETE FROM provinces WHERE id = ?", id)
	return err
}
