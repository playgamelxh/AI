package model

import "baokao/internal/db"

type MajorScore struct {
	ID             int    `json:"id" db:"id"`
	SchoolID       int    `json:"school_id" db:"school_id" binding:"required"`
	MajorID        int    `json:"major_id" db:"major_id" binding:"required"`
	ProvinceID     int    `json:"province_id" db:"province_id" binding:"required"`
	Year           int    `json:"year" db:"year" binding:"required"`
	LowestScore    int    `json:"lowest_score" db:"lowest_score" binding:"required"`
	AdmissionCount int    `json:"admission_count" db:"admission_count"`
	Batch          string `json:"batch" db:"batch"`
	Type           string `json:"type" db:"type"` // 理科, 文科, 综合, 物理类, 历史类
}

func GetMajorScores(schoolID int, provinceID int, year int) ([]MajorScore, error) {
	query := "SELECT id, school_id, major_id, province_id, year, lowest_score, admission_count, batch, type FROM major_scores WHERE 1=1"
	var args []interface{}

	if schoolID > 0 {
		query += " AND school_id = ?"
		args = append(args, schoolID)
	}
	if provinceID > 0 {
		query += " AND province_id = ?"
		args = append(args, provinceID)
	}
	if year > 0 {
		query += " AND year = ?"
		args = append(args, year)
	}
	
	query += " ORDER BY year DESC, lowest_score DESC"

	rows, err := db.DB.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []MajorScore
	for rows.Next() {
		var s MajorScore
		err := rows.Scan(&s.ID, &s.SchoolID, &s.MajorID, &s.ProvinceID, &s.Year, &s.LowestScore, &s.AdmissionCount, &s.Batch, &s.Type)
		if err != nil {
			continue
		}
		list = append(list, s)
	}
	if list == nil {
		list = []MajorScore{}
	}
	return list, nil
}

func CreateMajorScore(s *MajorScore) error {
	query := `INSERT INTO major_scores (school_id, major_id, province_id, year, lowest_score, admission_count, batch, type) 
	          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
	result, err := db.DB.Exec(query, s.SchoolID, s.MajorID, s.ProvinceID, s.Year, s.LowestScore, s.AdmissionCount, s.Batch, s.Type)
	if err != nil {
		return err
	}
	id, _ := result.LastInsertId()
	s.ID = int(id)
	return nil
}

func UpdateMajorScore(s *MajorScore) error {
	query := `UPDATE major_scores SET school_id=?, major_id=?, province_id=?, year=?, lowest_score=?, admission_count=?, batch=?, type=? WHERE id=?`
	_, err := db.DB.Exec(query, s.SchoolID, s.MajorID, s.ProvinceID, s.Year, s.LowestScore, s.AdmissionCount, s.Batch, s.Type, s.ID)
	return err
}

func DeleteMajorScore(id int) error {
	_, err := db.DB.Exec("DELETE FROM major_scores WHERE id = ?", id)
	return err
}
