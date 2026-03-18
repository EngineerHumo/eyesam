package onnxutil

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"rakrock/app/business/pkg/utils/tool"

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
		tool.AddLocalLibToPath()
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
	// 1. Check for CUDA environment variable before calling any GPU-related ORT functions
	cudaPath := os.Getenv("CUDA_PATH")

	if cudaPath != "" {
		fmt.Printf("CUDA_PATH detected: %s. Attempting to initialize GPU...\n", cudaPath)

		// 2. Try creating CUDA options only if hardware environment exists
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err == nil {
			err = opts.AppendExecutionProviderCUDA(cudaOpts)
			// Ensure cleanup of the C-level options object
			cudaOpts.Destroy()

			if err != nil {
				// Fallback if appending fails (e.g., driver version mismatch)
				fmt.Printf("Warning: Failed to append CUDA provider: %v. Falling back to CPU.\n", err)
			} else {
				fmt.Println("NVIDIA GPU acceleration enabled successfully.")
			}
		} else {
			// If NewCUDAProviderOptions fails (e.g., missing specific DLLs)
			fmt.Printf("Failed to create CUDA options: %v. Using CPU instead.\n", err)
		}
	} else {
		// 3. Skip GPU logic entirely if CUDA_PATH is missing
		fmt.Println("CUDA_PATH not found. Defaulting to CPU mode.")
	}
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
