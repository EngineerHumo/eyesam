package preprocess

import (
	"errors"

	"eyesam/standalone_app_v3_5_go/internal/utils"
)

var ErrNotImplemented = errors.New("pre-segmentation not implemented")

type Result struct {
	Labels  utils.Mask
	FAZMask utils.Mask
	AreaMask utils.Mask
}

type PreSegmenter struct {
	ModelPath string
}

func NewPreSegmenter(modelPath string) *PreSegmenter {
	return &PreSegmenter{ModelPath: modelPath}
}

func (p *PreSegmenter) Infer(_ utils.ModelImage) (Result, error) {
	return Result{}, ErrNotImplemented
}
