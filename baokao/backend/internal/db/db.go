package db

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/mysql"
	_ "github.com/golang-migrate/migrate/v4/source/file"
)

var DB *sql.DB

func InitDB(dsn string) error {
	var err error
	DB, err = sql.Open("mysql", dsn)
	if err != nil {
		return err
	}

	// Retry connection if db is not ready
	for i := 0; i < 10; i++ {
		err = DB.Ping()
		if err == nil {
			break
		}
		time.Sleep(2 * time.Second)
	}

	if err != nil {
		return fmt.Errorf("could not connect to db: %v", err)
	}
	
	DB.SetMaxOpenConns(10)
	DB.SetMaxIdleConns(5)
	
	return nil
}

func RunMigrations(dsn string) error {
	// Add multiStatements=true for migrations if not present
	if !strings.Contains(dsn, "multiStatements=true") {
		if strings.Contains(dsn, "?") {
			dsn += "&multiStatements=true"
		} else {
			dsn += "?multiStatements=true"
		}
	}

	migDb, err := sql.Open("mysql", dsn)
	if err != nil {
		return err
	}
	defer migDb.Close()

	driver, err := mysql.WithInstance(migDb, &mysql.Config{})
	if err != nil {
		return err
	}

	m, err := migrate.NewWithDatabaseInstance(
		"file://migrations",
		"mysql", driver)
	if err != nil {
		return err
	}

	if err := m.Up(); err != nil && err != migrate.ErrNoChange {
		return err
	}

	return nil
}
