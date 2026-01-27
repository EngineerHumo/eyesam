package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"eyesam/standalone_app_v3_5_go/internal/pipeline"
	"eyesam/standalone_app_v3_5_go/internal/ui"
)

func main() {
	log.SetFlags(0)
	baseDir := flag.String("base", ".", "base directory containing onnx models")
	flag.Parse()

	onnxDir := filepath.Join(*baseDir, "onnx")
	if err := ensureOnnxFiles(onnxDir); err != nil {
		log.Printf("%v", err)
		os.Exit(1)
	}

	pipe := pipeline.New(onnxDir)
	if err := ui.Run(pipe); err != nil {
		log.Printf("%v", err)
		os.Exit(1)
	}
}

func ensureOnnxFiles(onnxDir string) error {
	required := []string{"pre.onnx", "first.onnx", "iteration.onnx"}
	var missing []string
	for _, name := range required {
		if _, err := os.Stat(filepath.Join(onnxDir, name)); err != nil {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing ONNX files: %v (place them in %s)", missing, onnxDir)
	}
	return nil
}
