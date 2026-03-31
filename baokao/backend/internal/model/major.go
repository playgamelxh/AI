package model

import "baokao/internal/db"

type Major struct {
	ID          int    `json:"id" db:"id"`
	SchoolID    int    `json:"school_id" db:"school_id" binding:"required"`
	Name        string `json:"name" db:"name" binding:"required"`
	Description string `json:"description" db:"description"`
}

func GetMajorsBySchoolID(schoolID int) ([]Major, error) {
	query := "SELECT id, school_id, name, description FROM majors WHERE school_id = ? ORDER BY id DESC"
	rows, err := db.DB.Query(query, schoolID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []Major
	for rows.Next() {
		var m Major
		if err := rows.Scan(&m.ID, &m.SchoolID, &m.Name, &m.Description); err != nil {
			continue
		}
		list = append(list, m)
	}
	if list == nil {
		list = []Major{}
	}
	return list, nil
}

func CreateMajor(m *Major) error {
	result, err := db.DB.Exec("INSERT INTO majors (school_id, name, description) VALUES (?, ?, ?)", m.SchoolID, m.Name, m.Description)
	if err != nil {
		return err
	}
	id, _ := result.LastInsertId()
	m.ID = int(id)
	return nil
}

func UpdateMajor(m *Major) error {
	_, err := db.DB.Exec("UPDATE majors SET school_id = ?, name = ?, description = ? WHERE id = ?", m.SchoolID, m.Name, m.Description, m.ID)
	return err
}

func DeleteMajor(id int) error {
	_, err := db.DB.Exec("DELETE FROM majors WHERE id = ?", id)
	return err
}
