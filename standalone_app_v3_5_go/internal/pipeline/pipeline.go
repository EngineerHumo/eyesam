package pipeline

import (
	"fmt"

	"eyesam/standalone_app_v3_5_go/internal/inference"
	"eyesam/standalone_app_v3_5_go/internal/preprocess"
	"eyesam/standalone_app_v3_5_go/internal/utils"
)

type Result struct {
	SchemeMasks  []utils.Mask
	SchemeLogits []utils.Mask
	SchemeClicks []utils.Click
	CurrentLogits utils.Mask
	LastAutoClick utils.Click
	CurrentClick  utils.Click
	FAZCenter     [2]int
	AreaMask      utils.Mask
	FAZMask       utils.Mask
}

type Pipeline struct {
	PreModel      *preprocess.PreSegmenter
	FirstModel    *inference.OnnxModel
	IterationModel *inference.OnnxModel
}

func New(onnxDir string) *Pipeline {
	return &Pipeline{
		PreModel:       preprocess.NewPreSegmenter(fmt.Sprintf("%s/pre.onnx", onnxDir)),
		FirstModel:     inference.NewOnnxModel(fmt.Sprintf("%s/first.onnx", onnxDir)),
		IterationModel: inference.NewOnnxModel(fmt.Sprintf("%s/iteration.onnx", onnxDir)),
	}
}

func (p *Pipeline) RunInitial(_ utils.ModelImage) (Result, error) {
	return Result{}, fmt.Errorf("run initial: %w", inference.ErrNotImplemented)
}

func (p *Pipeline) RunIteration(_ utils.ModelImage, _ []utils.Click, _ utils.Mask) (inference.Result, error) {
	return inference.Result{}, fmt.Errorf("run iteration: %w", inference.ErrNotImplemented)
}
