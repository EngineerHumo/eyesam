package inference

import (
	"errors"
	"fmt"

	"eyesam/standalone_app_v3_5_go/internal/utils"
)

var ErrNotImplemented = errors.New("onnx inference not implemented")

type Result struct {
	Mask   utils.Mask
	Logits utils.Mask
}

type Model interface {
	ImageInputSize(fallbackWidth, fallbackHeight int) (int, int)
	Infer(image utils.ModelImage, clicks []utils.Click, maskInput *utils.Mask) (Result, error)
}

type OnnxModel struct {
	ModelPath string
}

func NewOnnxModel(path string) *OnnxModel {
	return &OnnxModel{ModelPath: path}
}

func (m *OnnxModel) ImageInputSize(fallbackWidth, fallbackHeight int) (int, int) {
	return fallbackWidth, fallbackHeight
}

func (m *OnnxModel) Infer(_ utils.ModelImage, _ []utils.Click, _ *utils.Mask) (Result, error) {
	return Result{}, fmt.Errorf("%w: model=%s", ErrNotImplemented, m.ModelPath)
}
