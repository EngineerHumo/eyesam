package pipeline

import (
	"fmt"
	"image"
	"strings"

	"eyesam/standalone_app_Tear_go_current_version/internal/inference"
	"eyesam/standalone_app_Tear_go_current_version/internal/preprocess"
	"eyesam/standalone_app_Tear_go_current_version/internal/utils"
)

type Pipeline struct {
	PreModel       *preprocess.PreSegmenter
	FirstModel     *inference.OnnxModel
	IterationModel *inference.OnnxModel
}

func New(onnxDir string) *Pipeline {
	return &Pipeline{
		PreModel:       preprocess.NewPreSegmenter(fmt.Sprintf("%s/pre.onnx", onnxDir)),
		FirstModel:     inference.NewOnnxModel(fmt.Sprintf("%s/first.onnx", onnxDir)),
		IterationModel: inference.NewOnnxModel(fmt.Sprintf("%s/iteration.onnx", onnxDir)),
	}
}

func (p *Pipeline) RunPreSegmentation(img image.Image) (utils.Mask, utils.Mask, error) {
	modelImage := utils.PrepareImageForModel(img, img.Bounds().Dx(), img.Bounds().Dy())
	result, err := p.PreModel.Infer(modelImage)
	if err != nil {
		return utils.Mask{}, utils.Mask{}, err
	}
	return result.AreaMask, result.FAZMask, nil
}

func (p *Pipeline) RunFirst(image utils.ModelImage, clicks []utils.Click) (inference.Result, error) {
	return p.FirstModel.Infer(image, clicks, nil)
}

func (p *Pipeline) RunIteration(image utils.ModelImage, clicks []utils.Click, prevLogits utils.FloatMask) (inference.Result, error) {
	if prevLogits.Width == 0 || prevLogits.Height == 0 {
		return inference.Result{}, fmt.Errorf("prev logits missing")
	}
	if err := p.IterationModel.EnsureInitialized(); err != nil {
		return inference.Result{}, err
	}
	resizedHW := [2]int{image.Resized.Bounds().Dy(), image.Resized.Bounds().Dx()}
	maskInputShape := p.findMaskInputShape(resizedHW)
	maskInput := utils.ResizeFloatMaskLinear(prevLogits, maskInputShape[1], maskInputShape[0])
	return p.IterationModel.Infer(image, clicks, &maskInput)
}

func (p *Pipeline) findMaskInputShape(resizedHW [2]int) [2]int {
	for name, shape := range p.IterationModel.InputShapes {
		if strings.Contains(name, "mask") && len(shape) == 4 && shape[1] == 1 {
			h := int(shape[2])
			w := int(shape[3])
			if h > 0 && w > 0 {
				return [2]int{h, w}
			}
		}
	}
	return [2]int{resizedHW[0] / 4, resizedHW[1] / 4}
}
