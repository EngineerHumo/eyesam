package standalone_app_v3_7_go

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"eyesam/standalone_app_v3_7_go/internal/inference"
	"eyesam/standalone_app_v3_7_go/internal/npy"
	"eyesam/standalone_app_v3_7_go/internal/pipeline"
	"eyesam/standalone_app_v3_7_go/internal/utils"
)

const (
	maskTemplate       = "mask_%d.npy"
	logitsTemplate     = "logits_%d.npy"
	clicksTemplate     = "clicks_%d.json"
	currentMaskFile    = "current_mask.npy"
	currentLogitsFile  = "current_logits.npy"
	currentClicksFile  = "current_clicks.json"
	planningCenterFile = "planning_center.json"
	previewMaskFile    = "preview_mask.npy"
)

type PlanArtifacts struct {
	Masks          []string `json:"masks"`
	Logits         []string `json:"logits"`
	Clicks         []string `json:"clicks"`
	PlanningCenter string   `json:"planning_center"`
}

type Click struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Label int     `json:"label"`
}

type clickPayload struct {
	Points []Click `json:"points"`
}

type SurgicalInterface struct {
	pipeline     *pipeline.Pipeline
	imagePath    string
	image        image.Image
	modelImage   *utils.ModelImage
	areaMask     *utils.Mask
	fazMask      *utils.Mask
	fazCenter    image.Point
	outputDir    string
	currentIndex *int
}

func NewSurgicalInterface(onnxDir string) (*SurgicalInterface, error) {
	if err := ensureOnnxFiles(onnxDir); err != nil {
		return nil, err
	}
	return &SurgicalInterface{pipeline: pipeline.New(onnxDir)}, nil
}

func ensureOnnxFiles(onnxDir string) error {
	required := []string{"pre.onnx", "first.onnx", "iteration.onnx"}
	missing := []string{}
	for _, name := range required {
		path := filepath.Join(onnxDir, name)
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				missing = append(missing, name)
				continue
			}
			return fmt.Errorf("检查 ONNX 文件 %s 失败: %w", name, err)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("缺少 ONNX 文件: %v，请确认 %s 中包含模型", missing, onnxDir)
	}
	return nil
}

func (s *SurgicalInterface) setImage(imagePath string) error {
	s.imagePath = imagePath
	img, err := utils.LoadImage(imagePath)
	if err != nil {
		return err
	}
	firstW, firstH := s.pipeline.FirstModel.ImageInputSize(img.Bounds().Dx(), img.Bounds().Dy())
	modelImage := utils.PrepareImageForModel(img, firstW, firstH)
	s.image = img
	s.modelImage = &modelImage
	return nil
}

func (s *SurgicalInterface) ensureOutputDir(outputDir string) error {
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return err
	}
	s.outputDir = outputDir
	return nil
}

func (s *SurgicalInterface) maskPath(index int) string {
	return filepath.Join(s.outputDir, fmt.Sprintf(maskTemplate, index))
}

func (s *SurgicalInterface) logitsPath(index int) string {
	return filepath.Join(s.outputDir, fmt.Sprintf(logitsTemplate, index))
}

func (s *SurgicalInterface) clicksPath(index int) string {
	return filepath.Join(s.outputDir, fmt.Sprintf(clicksTemplate, index))
}

func (s *SurgicalInterface) currentMaskPath() string {
	return filepath.Join(s.outputDir, currentMaskFile)
}

func (s *SurgicalInterface) currentLogitsPath() string {
	return filepath.Join(s.outputDir, currentLogitsFile)
}

func (s *SurgicalInterface) currentClicksPath() string {
	return filepath.Join(s.outputDir, currentClicksFile)
}

func (s *SurgicalInterface) planningCenterPath() string {
	return filepath.Join(s.outputDir, planningCenterFile)
}

func (s *SurgicalInterface) previewMaskPath() string {
	return filepath.Join(s.outputDir, previewMaskFile)
}

func (s *SurgicalInterface) resolveCurrentPaths() (string, string, string, error) {
	if s.outputDir == "" {
		return "", "", "", fmt.Errorf("尚未设置输出目录，请先调用初始规划或提供输出路径")
	}
	if s.currentIndex == nil {
		return s.currentMaskPath(), s.currentLogitsPath(), s.currentClicksPath(), nil
	}
	idx := *s.currentIndex
	return s.maskPath(idx), s.logitsPath(idx), s.clicksPath(idx), nil
}

func (s *SurgicalInterface) saveClicks(clicks []Click, path string) error {
	payload := clickPayload{Points: clicks}
	return writeJSON(path, payload)
}

func (s *SurgicalInterface) loadClicks(path string) ([]Click, error) {
	if !fileExists(path) {
		return nil, fmt.Errorf("clicks 文件不存在: %s", path)
	}
	var rawPayload map[string]any
	if err := readJSON(path, &rawPayload); err == nil {
		if points, ok := rawPayload["points"]; ok {
			clicks, err := parseClickList(points)
			if err != nil {
				return nil, err
			}
			return clicks, nil
		}
	}
	var rawList any
	if err := readJSON(path, &rawList); err == nil {
		clicks, err := parseClickList(rawList)
		if err != nil {
			return nil, err
		}
		if len(clicks) > 0 {
			return clicks, nil
		}
	}
	return []Click{}, nil
}

func parseClickList(payload any) ([]Click, error) {
	items, ok := payload.([]any)
	if !ok {
		return []Click{}, nil
	}
	clicks := make([]Click, 0, len(items))
	for _, item := range items {
		point, ok := item.(map[string]any)
		if !ok {
			continue
		}
		x, okX := point["x"].(float64)
		y, okY := point["y"].(float64)
		if !okX || !okY {
			continue
		}
		label := 1
		if rawLabel, okLabel := point["label"]; okLabel {
			if v, ok := rawLabel.(float64); ok {
				label = int(v)
			}
		}
		clicks = append(clicks, Click{X: x, Y: y, Label: label})
	}
	return clicks, nil
}

func toModelClicks(clicks []Click) []utils.Click {
	out := make([]utils.Click, 0, len(clicks))
	for _, c := range clicks {
		out = append(out, utils.Click{X: c.X, Y: c.Y, Label: c.Label})
	}
	return out
}

func (s *SurgicalInterface) InitialPlan(imagePath, outputDir string) (*PlanArtifacts, error) {
	if err := s.setImage(imagePath); err != nil {
		return nil, err
	}
	if err := s.ensureOutputDir(outputDir); err != nil {
		return nil, err
	}
	result, err := s.pipeline.RunInitial(s.image, nil)
	if err != nil {
		return nil, err
	}

	s.areaMask = &result.AreaMask
	s.fazMask = &result.FAZMask
	s.fazCenter = result.FAZCenter
	s.currentIndex = nil

	masks := make([]string, 0, len(result.SchemeMasks))
	logits := make([]string, 0, len(result.SchemeLogits))
	clicks := make([]string, 0, len(result.SchemeClicks))
	for idx := range result.SchemeMasks {
		maskPath := s.maskPath(idx)
		logitsPath := s.logitsPath(idx)
		clicksPath := s.clicksPath(idx)
		if err := npy.WriteUint8(maskPath, result.SchemeMasks[idx]); err != nil {
			return nil, err
		}
		if err := npy.WriteFloat32(logitsPath, result.SchemeLogits[idx]); err != nil {
			return nil, err
		}
		clickList := []Click{}
		if idx < len(result.SchemeClicks) {
			click := result.SchemeClicks[idx]
			clickList = append(clickList, Click{X: click.X, Y: click.Y, Label: click.Label})
		}
		if err := s.saveClicks(clickList, clicksPath); err != nil {
			return nil, err
		}
		masks = append(masks, maskPath)
		logits = append(logits, logitsPath)
		clicks = append(clicks, clicksPath)
	}

	centerPath := s.planningCenterPath()
	if err := writeJSON(centerPath, map[string]int{"x": s.fazCenter.X, "y": s.fazCenter.Y}); err != nil {
		return nil, err
	}

	return &PlanArtifacts{
		Masks:          masks,
		Logits:         logits,
		Clicks:         clicks,
		PlanningCenter: centerPath,
	}, nil
}

func (s *SurgicalInterface) SelectInitialScheme(maskIndex int) error {
	if s.outputDir == "" {
		return fmt.Errorf("尚未设置输出目录，请先调用初始规划")
	}
	maskPath := s.maskPath(maskIndex)
	logitsPath := s.logitsPath(maskIndex)
	clicksPath := s.clicksPath(maskIndex)
	if !fileExists(maskPath) || !fileExists(logitsPath) || !fileExists(clicksPath) {
		return fmt.Errorf("指定的 mask/logits/clicks 文件不存在，请检查 mask 编号")
	}

	maskFiles, _ := filepath.Glob(filepath.Join(s.outputDir, "mask_*.npy"))
	for _, path := range maskFiles {
		if path == maskPath {
			continue
		}
		_ = os.Remove(path)
	}
	logitsFiles, _ := filepath.Glob(filepath.Join(s.outputDir, "logits_*.npy"))
	for _, path := range logitsFiles {
		if path == logitsPath {
			continue
		}
		_ = os.Remove(path)
	}
	clickFiles, _ := filepath.Glob(filepath.Join(s.outputDir, "clicks_*.json"))
	for _, path := range clickFiles {
		if path == clicksPath {
			continue
		}
		_ = os.Remove(path)
	}

	s.currentIndex = &maskIndex
	return nil
}

func (s *SurgicalInterface) ApplyClicks(clicksJSON string) (map[string]string, error) {
	if s.image == nil || s.modelImage == nil {
		return nil, fmt.Errorf("尚未加载图像，请先调用初始规划")
	}
	if s.outputDir == "" {
		return nil, fmt.Errorf("尚未设置输出目录，请先调用初始规划")
	}
	if s.areaMask == nil {
		return nil, fmt.Errorf("缺少区域掩码，请先执行初始规划")
	}

	newClicks, err := s.loadClicks(clicksJSON)
	if err != nil {
		return nil, err
	}
	maskPath, logitsPath, clicksPath, err := s.resolveCurrentPaths()
	if err != nil {
		return nil, err
	}
	merged := newClicks
	if fileExists(clicksPath) {
		existing, err := s.loadClicks(clicksPath)
		if err != nil {
			return nil, err
		}
		merged = append(existing, newClicks...)
	}

	clicks := toModelClicks(merged)
	var result inference.Result
	var displayMask utils.Mask
	if fileExists(logitsPath) && fileExists(maskPath) {
		prevLogits, err := npy.ReadFloat32(logitsPath)
		if err != nil {
			return nil, err
		}
		iterResult, err := s.pipeline.RunIteration(*s.modelImage, clicks, prevLogits)
		if err != nil {
			return nil, err
		}
		result = iterResult
		displayMask = utils.ResizeMaskNearest(result.Mask, s.image.Bounds().Dx(), s.image.Bounds().Dy())
		displayMask = applyAreaConstraint(displayMask, *s.areaMask)
	} else {
		firstResult, err := s.pipeline.FirstModel.Infer(*s.modelImage, clicks, nil)
		if err != nil {
			return nil, err
		}
		result = firstResult
		displayMask = utils.ResizeMaskNearest(result.Mask, s.image.Bounds().Dx(), s.image.Bounds().Dy())
		displayMask = postprocessFirstMask(displayMask)
		displayMask = applyAreaConstraint(displayMask, *s.areaMask)
	}

	if err := npy.WriteUint8(maskPath, displayMask); err != nil {
		return nil, err
	}
	if err := npy.WriteFloat32(logitsPath, result.Logits); err != nil {
		return nil, err
	}
	if err := s.saveClicks(merged, clicksPath); err != nil {
		return nil, err
	}

	return map[string]string{
		"mask":   maskPath,
		"logits": logitsPath,
		"clicks": clicksPath,
	}, nil
}

func (s *SurgicalInterface) PreviewClicks(clicksJSON string) (string, error) {
	if s.image == nil || s.modelImage == nil {
		return "", fmt.Errorf("尚未加载图像，请先调用初始规划")
	}
	if s.outputDir == "" {
		return "", fmt.Errorf("尚未设置输出目录，请先调用初始规划")
	}
	if s.areaMask == nil {
		return "", fmt.Errorf("缺少区域掩码，请先执行初始规划")
	}

	newClicks, err := s.loadClicks(clicksJSON)
	if err != nil {
		return "", err
	}
	maskPath, logitsPath, clicksPath, err := s.resolveCurrentPaths()
	if err != nil {
		return "", err
	}
	merged := newClicks
	if fileExists(clicksPath) {
		existing, err := s.loadClicks(clicksPath)
		if err != nil {
			return "", err
		}
		merged = append(existing, newClicks...)
	}

	clicks := toModelClicks(merged)
	var displayMask utils.Mask
	if fileExists(logitsPath) && fileExists(maskPath) {
		prevLogits, err := npy.ReadFloat32(logitsPath)
		if err != nil {
			return "", err
		}
		iterResult, err := s.pipeline.RunIteration(*s.modelImage, clicks, prevLogits)
		if err != nil {
			return "", err
		}
		displayMask = utils.ResizeMaskNearest(iterResult.Mask, s.image.Bounds().Dx(), s.image.Bounds().Dy())
		displayMask = applyAreaConstraint(displayMask, *s.areaMask)
	} else {
		firstResult, err := s.pipeline.FirstModel.Infer(*s.modelImage, clicks, nil)
		if err != nil {
			return "", err
		}
		displayMask = utils.ResizeMaskNearest(firstResult.Mask, s.image.Bounds().Dx(), s.image.Bounds().Dy())
		displayMask = postprocessFirstMask(displayMask)
		displayMask = applyAreaConstraint(displayMask, *s.areaMask)
	}

	previewPath := s.previewMaskPath()
	if err := npy.WriteUint8(previewPath, displayMask); err != nil {
		return "", err
	}
	return previewPath, nil
}

func (s *SurgicalInterface) ClearCurrentPlan() error {
	if s.outputDir == "" {
		return nil
	}
	maskPath, logitsPath, clicksPath, err := s.resolveCurrentPaths()
	if err != nil {
		return err
	}
	_ = os.Remove(maskPath)
	_ = os.Remove(logitsPath)
	_ = os.Remove(clicksPath)
	s.currentIndex = nil
	return nil
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

func writeJSON(path string, payload any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
