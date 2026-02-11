package onnxutil

import (
	"fmt"
	"os"
	"runtime"
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
		} else {
			path = GetSharedLibPath()
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
	} else {
		if err := opts.AppendExecutionProviderDirectML(0); err != nil {
			return nil, fmt.Errorf("append DirectML provider: %w", err)
		}
	}
	// cudaOpts, err := ort.NewCUDAProviderOptions()
	// if err == nil {
	// 	err = opts.AppendExecutionProviderCUDA(cudaOpts)
	// 	cudaOpts.Destroy()
	// 	if err != nil {
	// 		return nil, fmt.Errorf("append cuda provider: %w", err)
	// 	}
	// } else {
	// 	fmt.Println("init CUDA", err)
	// 	fmt.Println("try to init intel GPU")
	// 	if err := opts.AppendExecutionProviderDirectML(0); err != nil {
	// 		return nil, fmt.Errorf("append DirectML provider: %w", err)
	// 	}
	// }
	return opts, nil
}

func Shutdown() {
	if ort.IsInitialized() {
		ort.DestroyEnvironment()
	}
}

func GetSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return ".\\lib\\onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./lib/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "./lib/onnxruntime_amd64.dylib"
		}

	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./lib/onnxruntime_arm64.so"
		}
		return "./lib/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}
