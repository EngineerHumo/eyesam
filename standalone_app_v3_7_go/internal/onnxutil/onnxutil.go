package onnxutil

import (
	"fmt"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var initOnce sync.Once
var initErr error

func InitializeEnvironment() error {
	initOnce.Do(func() {
		if path := os.Getenv("ORT_SHARED_LIBRARY_PATH"); path != "" {
			ort.SetSharedLibraryPath(path)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("initialize onnxruntime: %w", err)
			return
		}
	})
	return initErr
}

func SessionOptions() (*ort.SessionOptions, error) {
	if err := InitializeEnvironment(); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if useCUDA := strings.EqualFold(os.Getenv("EYESAM_ONNX_USE_CUDA"), "1") || strings.EqualFold(os.Getenv("EYESAM_ONNX_USE_CUDA"), "true"); useCUDA {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			err = opts.AppendExecutionProviderCUDA(cudaOpts)
			cudaOpts.Destroy()
			if err != nil {
				return nil, fmt.Errorf("append cuda provider: %w", err)
			}
		}
	}
	return opts, nil
}

func Shutdown() {
	if ort.IsInitialized() {
		ort.DestroyEnvironment()
	}
}
