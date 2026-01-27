package pipeline

import (
	"fmt"
	"image"
	"log"
	"strings"

	"eyesam/standalone_app_v3_5_go/internal/inference"
	"eyesam/standalone_app_v3_5_go/internal/planner"
	"eyesam/standalone_app_v3_5_go/internal/preprocess"
	"eyesam/standalone_app_v3_5_go/internal/utils"
)

type Result struct {
	SchemeMasks   []utils.Mask
	SchemeLogits  []utils.FloatMask
	SchemeClicks  []utils.Click
	CurrentLogits utils.FloatMask
	LastAutoClick utils.Click
	CurrentClick  utils.Click
	FAZCenter     image.Point
	AreaMask      utils.Mask
	FAZMask       utils.Mask
}

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

func (p *Pipeline) RunInitial(img image.Image, progress func(int)) (Result, error) {
	firstW, firstH := p.FirstModel.ImageInputSize(img.Bounds().Dx(), img.Bounds().Dy())
	firstImage := utils.PrepareImageForModel(img, firstW, firstH)

	preResult, err := p.PreModel.Infer(firstImage)
	if err != nil {
		return Result{}, err
	}
	fazMask := preResult.FAZMask
	areaMask := preResult.AreaMask

	areaBin := utils.Binarize(areaMask, 1)
	areaLcc := utils.LargestConnectedComponent(areaBin)
	click0 := utils.InscribedCenter(areaLcc)
	log.Printf("auto_click0=(%d,%d)", click0.X, click0.Y)

	resizedHW := [2]int{firstImage.Resized.Bounds().Dy(), firstImage.Resized.Bounds().Dx()}
	firstResult, lastClick, err := p.runFirstWithClick(firstImage, resizedHW, click0)
	if err != nil {
		return Result{}, err
	}
	runCount := 1
	if progress != nil {
		progress(runCount)
	}

	currentResult := firstResult
	currentClick := click0
	for idx := 0; idx < 4; idx++ {
		prevBin := utils.Binarize(currentResult.Mask, 1)
		prevLcc := utils.LargestConnectedComponent(prevBin)
		center := utils.InscribedCenter(prevLcc)
		maskH := currentResult.Mask.Height
		maskW := currentResult.Mask.Width
		scaleX := float64(firstImage.Original.Bounds().Dx()) / float64(maskW)
		scaleY := float64(firstImage.Original.Bounds().Dy()) / float64(maskH)
		currentClick = image.Point{X: int(float64(center.X) * scaleX), Y: int(float64(center.Y) * scaleY)}
		log.Printf("auto_click%d=(%d,%d)", idx+1, currentClick.X, currentClick.Y)
		currentResult, lastClick, err = p.runFirstWithClick(firstImage, resizedHW, currentClick)
		if err != nil {
			return Result{}, err
		}
		runCount++
		if progress != nil {
			progress(runCount)
		}
	}

	fazCenter := planner.ComputeFAZCenter(fazMask)
	displayMask := utils.ResizeMaskNearest(currentResult.Mask, img.Bounds().Dx(), img.Bounds().Dy())
	displayMask = postprocessFirstMask(displayMask)
	displayMask = applyAreaConstraint(displayMask, areaMask)

	areaTotal := utils.MaskArea(areaBin)
	schemeMasks := []utils.Mask{displayMask}
	schemeLogits := []utils.FloatMask{currentResult.Logits}
	schemeClicks := []utils.Click{lastClick}
	schemeUnion := displayMask.Clone()
	coverage := 0.0
	if areaTotal > 0 {
		coverage = float64(utils.MaskArea(schemeUnion)) / float64(areaTotal)
	}
	if coverage >= 0.9 {
		schemeMasks[0] = areaBin
		schemeUnion = areaBin.Clone()
		coverage = 1.0
	} else {
		rejected := utils.NewMask(areaBin.Width, areaBin.Height)
		for runCount < 15 && coverage < 0.9 {
			remaining := utils.AndMask(areaBin, utils.NotMask(schemeUnion))
			remaining = utils.AndMask(remaining, utils.NotMask(rejected))
			if utils.MaskArea(remaining) == 0 {
				break
			}
			center := utils.InscribedCenter(remaining)
			log.Printf("auto_scheme_click=(%d,%d)", center.X, center.Y)
			candidate, candidateClick, err := p.runFirstWithClick(firstImage, resizedHW, center)
			if err != nil {
				return Result{}, err
			}
			runCount++
			if progress != nil {
				progress(runCount)
			}
			candidateMask := utils.ResizeMaskNearest(candidate.Mask, img.Bounds().Dx(), img.Bounds().Dy())
			candidateMask = postprocessFirstMask(candidateMask)
			candidateMask = applyAreaConstraint(candidateMask, areaMask)
			if isValidScheme(candidateMask, schemeMasks) {
				schemeMasks = append(schemeMasks, candidateMask)
				schemeLogits = append(schemeLogits, candidate.Logits)
				schemeClicks = append(schemeClicks, candidateClick)
				schemeUnion = utils.OrMask(schemeUnion, candidateMask)
				if areaTotal > 0 {
					coverage = float64(utils.MaskArea(schemeUnion)) / float64(areaTotal)
				}
			} else {
				rejected = utils.OrMask(rejected, candidateMask)
			}
		}
	}

	return Result{
		SchemeMasks:   schemeMasks,
		SchemeLogits:  schemeLogits,
		SchemeClicks:  schemeClicks,
		CurrentLogits: currentResult.Logits,
		LastAutoClick: lastClick,
		CurrentClick:  utils.Click{X: float64(currentClick.X), Y: float64(currentClick.Y), Label: 1},
		FAZCenter:     fazCenter,
		AreaMask:      areaMask,
		FAZMask:       fazMask,
	}, nil
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

func (p *Pipeline) runFirstWithClick(firstImage utils.ModelImage, resizedHW [2]int, click image.Point) (inference.Result, utils.Click, error) {
	clickData := utils.Click{X: float64(click.X), Y: float64(click.Y), Label: 1}
	log.Printf("auto_click=(%d,%d)", click.X, click.Y)
	result, err := p.FirstModel.Infer(firstImage, []utils.Click{clickData}, nil)
	if err != nil {
		return inference.Result{}, utils.Click{}, err
	}
	return result, clickData, nil
}

func postprocessFirstMask(mask utils.Mask) utils.Mask {
	cleaned := utils.RemoveSmallComponents(mask, 600)
	filled := utils.FillSmallHoles(cleaned, 400)
	return filled
}

func applyAreaConstraint(mask utils.Mask, areaMask utils.Mask) utils.Mask {
	areaBin := utils.Binarize(areaMask, 1)
	return utils.AndMask(mask, areaBin)
}

func maskArea(mask utils.Mask) int {
	return utils.MaskArea(mask)
}

func isValidScheme(candidate utils.Mask, existing []utils.Mask) bool {
	candidateArea := maskArea(candidate)
	if candidateArea == 0 {
		return false
	}
	for _, mask := range existing {
		existingArea := maskArea(mask)
		if existingArea == 0 {
			continue
		}
		intersection := utils.MaskArea(utils.AndMask(candidate, mask))
		if intersection >= existingArea/2 {
			return false
		}
		if intersection >= candidateArea/2 {
			return false
		}
		ratio := float64(existingArea) / float64(candidateArea)
		if ratio <= 0.2 || ratio >= 5 {
			return false
		}
	}
	return true
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
