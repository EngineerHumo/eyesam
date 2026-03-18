package standalone_app_Tear_go_current_version

import (
	"fmt"
	"image"
	"os"
	"path/filepath"

	"eyesam/standalone_app_Tear_go_current_version/internal/inference"
	"eyesam/standalone_app_Tear_go_current_version/internal/pipeline"
	"eyesam/standalone_app_Tear_go_current_version/internal/planner"
	"eyesam/standalone_app_Tear_go_current_version/internal/utils"
)

type Service struct {
	pipeline *pipeline.Pipeline
}

type Click struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Label int     `json:"label"`
}

type PreSegmentationResult struct {
	AreaMask utils.Mask
	FAZMask  utils.Mask
}

type IterationResult struct {
	Mask   utils.Mask
	Logits utils.FloatMask
}

type PlanRequest struct {
	Image        image.Image
	Mask         utils.Mask
	AreaMask     *utils.Mask
	FAZMask      *utils.Mask
	SpotDiameter int
	SpotDistance int
	MaxLayers    int
}

func NewService(onnxDir string) (*Service, error) {
	if err := ensureOnnxFiles(onnxDir); err != nil {
		return nil, err
	}
	return &Service{pipeline: pipeline.New(onnxDir)}, nil
}

func ensureOnnxFiles(onnxDir string) error {
	required := []string{"pre.onnx", "first.onnx", "iteration.onnx"}
	missing := make([]string, 0)
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

func LoadImage(path string) (image.Image, error) {
	return utils.LoadImage(path)
}

func PrepareModelImage(img image.Image, targetWidth, targetHeight int) utils.ModelImage {
	return utils.PrepareImageForModel(img, targetWidth, targetHeight)
}

func ToModelClicks(clicks []Click) []utils.Click {
	out := make([]utils.Click, 0, len(clicks))
	for _, c := range clicks {
		out = append(out, utils.Click{X: c.X, Y: c.Y, Label: c.Label})
	}
	return out
}

func (s *Service) RunPreSegmentation(img image.Image) (*PreSegmentationResult, error) {
	areaMask, fazMask, err := s.pipeline.RunPreSegmentation(img)
	if err != nil {
		return nil, err
	}
	return &PreSegmentationResult{AreaMask: areaMask, FAZMask: fazMask}, nil
}

func (s *Service) RunFirst(img utils.ModelImage, clicks []Click) (*IterationResult, error) {
	result, err := s.pipeline.RunFirst(img, ToModelClicks(clicks))
	if err != nil {
		return nil, err
	}
	return &IterationResult{Mask: result.Mask, Logits: result.Logits}, nil
}

func (s *Service) RunIteration(img utils.ModelImage, clicks []Click, prevLogits utils.FloatMask) (*IterationResult, error) {
	result, err := s.pipeline.RunIteration(img, ToModelClicks(clicks), prevLogits)
	if err != nil {
		return nil, err
	}
	return &IterationResult{Mask: result.Mask, Logits: result.Logits}, nil
}

func (s *Service) PlanSurgery(req PlanRequest) utils.PlanResult {
	spotDiameter := req.SpotDiameter
	if spotDiameter <= 0 {
		spotDiameter = utils.DefaultSpotDiameter
	}
	spotDistance := req.SpotDistance
	if spotDistance < 0 {
		spotDistance = utils.DefaultSpotDistance
	}
	maxLayers := req.MaxLayers
	if maxLayers <= 0 {
		maxLayers = 3
	}
	return planner.PlanSurgery(req.Image, req.Mask, req.AreaMask, req.FAZMask, spotDiameter, spotDistance, maxLayers)
}

func ApplyAreaConstraint(mask, areaMask utils.Mask) utils.Mask {
	return utils.AndMask(mask, utils.Binarize(areaMask, 1))
}

func ResizeMaskToImage(mask utils.Mask, img image.Image) utils.Mask {
	return utils.ResizeMaskNearest(mask, img.Bounds().Dx(), img.Bounds().Dy())
}

func ResultFromInference(result inference.Result) IterationResult {
	return IterationResult{Mask: result.Mask, Logits: result.Logits}
}
