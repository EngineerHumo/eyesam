package ui

import (
	"fmt"
	"log"

	"eyesam/standalone_app_v3_5_go/internal/pipeline"
)

func Run(_ *pipeline.Pipeline) error {
	log.Println("UI not implemented in Go version yet")
	return fmt.Errorf("ui not implemented")
}
