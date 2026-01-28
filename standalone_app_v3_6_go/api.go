package standalone_app_v3_6_go

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"eyesam/standalone_app_v3_6_go/internal/inference"
	"eyesam/standalone_app_v3_6_go/internal/npy"
	"eyesam/standalone_app_v3_6_go/internal/pipeline"
	"eyesam/standalone_app_v3_6_go/internal/utils"
)

const (
	stateFile          = "state.json"
	areaMaskFile       = "area_mask.npy"
	fazMaskFile        = "faz_mask.npy"
	planningCenterFile = "planning_center.json"
	currentMaskFile    = "current_mask.npy"
	currentLogitsFile  = "current_logits.npy"
	currentClicksFile  = "current_clicks.json"
)

type Planner struct {
	pipeline *pipeline.Pipeline
}

type PlanningCenter struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type State struct {
	ImagePath   string `json:"image_path"`
	ImageWidth  int    `json:"image_width"`
	ImageHeight int    `json:"image_height"`
}

type Click struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Label int     `json:"label"`
}

type ClickPayload struct {
	Clicks []Click `json:"clicks"`
}

type Point struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type PolygonPayload struct {
	Points []Point `json:"points"`
}

type InitialScheme struct {
	MaskPath   string `json:"mask"`
	LogitsPath string `json:"logits"`
	ClicksPath string `json:"clicks"`
}

type InitialPlanResult struct {
	PlanningCenterPath string          `json:"planning_center"`
	Schemes            []InitialScheme `json:"schemes"`
}

func NewPlanner(onnxDir string) *Planner {
	return &Planner{pipeline: pipeline.New(onnxDir)}
}

func (p *Planner) InitialPlan(imagePath, workDir string) (*InitialPlanResult, error) {
	if err := ensureDir(workDir); err != nil {
		return nil, err
	}
	if err := clearSchemeFiles(workDir); err != nil {
		return nil, err
	}
	if err := clearCurrentFiles(workDir); err != nil {
		return nil, err
	}

	img, err := utils.LoadImage(imagePath)
	if err != nil {
		return nil, err
	}
	result, err := p.pipeline.RunInitial(img, nil)
	if err != nil {
		return nil, err
	}

	schemes := make([]InitialScheme, 0, len(result.SchemeMasks))
	for idx, mask := range result.SchemeMasks {
		maskPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_mask.npy", idx))
		logitsPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_logits.npy", idx))
		clicksPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_clicks.json", idx))
		if err := npy.WriteUint8(maskPath, mask); err != nil {
			return nil, err
		}
		if err := npy.WriteFloat32(logitsPath, result.SchemeLogits[idx]); err != nil {
			return nil, err
		}
		clicks := []Click{}
		if idx < len(result.SchemeClicks) {
			click := result.SchemeClicks[idx]
			clicks = append(clicks, Click{X: click.X, Y: click.Y, Label: click.Label})
		}
		if err := writeJSON(clicksPath, ClickPayload{Clicks: clicks}); err != nil {
			return nil, err
		}
		schemes = append(schemes, InitialScheme{
			MaskPath:   maskPath,
			LogitsPath: logitsPath,
			ClicksPath: clicksPath,
		})
	}

	if err := npy.WriteUint8(filepath.Join(workDir, areaMaskFile), result.AreaMask); err != nil {
		return nil, err
	}
	if err := npy.WriteUint8(filepath.Join(workDir, fazMaskFile), result.FAZMask); err != nil {
		return nil, err
	}

	centerPath := filepath.Join(workDir, planningCenterFile)
	if err := writeJSON(centerPath, PlanningCenter{X: result.FAZCenter.X, Y: result.FAZCenter.Y}); err != nil {
		return nil, err
	}
	statePath := filepath.Join(workDir, stateFile)
	state := State{
		ImagePath:   absPath(imagePath),
		ImageWidth:  img.Bounds().Dx(),
		ImageHeight: img.Bounds().Dy(),
	}
	if err := writeJSON(statePath, state); err != nil {
		return nil, err
	}

	return &InitialPlanResult{
		PlanningCenterPath: centerPath,
		Schemes:            schemes,
	}, nil
}

func (p *Planner) SelectInitialScheme(workDir string, schemeIndex int) error {
	maskPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_mask.npy", schemeIndex))
	logitsPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_logits.npy", schemeIndex))
	clicksPath := filepath.Join(workDir, fmt.Sprintf("scheme_%d_clicks.json", schemeIndex))
	if !fileExists(maskPath) || !fileExists(logitsPath) {
		return fmt.Errorf("未找到编号为 %d 的初始方案", schemeIndex)
	}

	indices, err := listSchemeIndices(workDir)
	if err != nil {
		return err
	}
	for _, idx := range indices {
		if idx == schemeIndex {
			continue
		}
		_ = os.Remove(filepath.Join(workDir, fmt.Sprintf("scheme_%d_mask.npy", idx)))
		_ = os.Remove(filepath.Join(workDir, fmt.Sprintf("scheme_%d_logits.npy", idx)))
		_ = os.Remove(filepath.Join(workDir, fmt.Sprintf("scheme_%d_clicks.json", idx)))
	}
	if err := copyFile(maskPath, filepath.Join(workDir, currentMaskFile)); err != nil {
		return err
	}
	if err := copyFile(logitsPath, filepath.Join(workDir, currentLogitsFile)); err != nil {
		return err
	}
	if fileExists(clicksPath) {
		if err := copyFile(clicksPath, filepath.Join(workDir, currentClicksFile)); err != nil {
			return err
		}
	} else {
		if err := writeJSON(filepath.Join(workDir, currentClicksFile), ClickPayload{Clicks: []Click{}}); err != nil {
			return err
		}
	}
	return nil
}

func (p *Planner) IterateWithClicks(workDir, clickJSONPath string) (string, string, string, error) {
	state, err := loadState(workDir)
	if err != nil {
		return "", "", "", err
	}
	img, err := utils.LoadImage(state.ImagePath)
	if err != nil {
		return "", "", "", err
	}
	modelW, modelH := p.pipeline.IterationModel.ImageInputSize(img.Bounds().Dx(), img.Bounds().Dy())
	modelImage := utils.PrepareImageForModel(img, modelW, modelH)

	newClicks, err := loadClicks(clickJSONPath, true)
	if err != nil {
		return "", "", "", err
	}
	existingClicks, err := loadClicks(filepath.Join(workDir, currentClicksFile), false)
	if err != nil {
		return "", "", "", err
	}
	combined := append(existingClicks, newClicks...)
	clicks := toModelClicks(combined)

	var result inference.Result
	var mask utils.Mask
	areaMask, err := loadAreaMask(workDir)
	if err != nil {
		return "", "", "", err
	}
	currentLogitsPath := filepath.Join(workDir, currentLogitsFile)
	if fileExists(currentLogitsPath) {
		prevLogits, err := npy.ReadFloat32(currentLogitsPath)
		if err != nil {
			return "", "", "", err
		}
		result, err = p.pipeline.RunIteration(modelImage, clicks, prevLogits)
		if err != nil {
			return "", "", "", err
		}
		mask = utils.ResizeMaskNearest(result.Mask, img.Bounds().Dx(), img.Bounds().Dy())
		mask = applyAreaConstraint(mask, areaMask)
	} else {
		firstW, firstH := p.pipeline.FirstModel.ImageInputSize(img.Bounds().Dx(), img.Bounds().Dy())
		firstImage := utils.PrepareImageForModel(img, firstW, firstH)
		firstResult, err := p.pipeline.FirstModel.Infer(firstImage, clicks, nil)
		if err != nil {
			return "", "", "", err
		}
		result = firstResult
		mask = utils.ResizeMaskNearest(result.Mask, img.Bounds().Dx(), img.Bounds().Dy())
		mask = postprocessFirstMask(mask, areaMask)
	}

	maskPath := filepath.Join(workDir, currentMaskFile)
	logitsPath := filepath.Join(workDir, currentLogitsFile)
	clicksPath := filepath.Join(workDir, currentClicksFile)
	if err := npy.WriteUint8(maskPath, mask); err != nil {
		return "", "", "", err
	}
	if err := npy.WriteFloat32(logitsPath, result.Logits); err != nil {
		return "", "", "", err
	}
	if err := writeJSON(clicksPath, ClickPayload{Clicks: combined}); err != nil {
		return "", "", "", err
	}

	return maskPath, logitsPath, clicksPath, nil
}

func (p *Planner) AddClickPoint(workDir, clickJSONPath string) (string, error) {
	return updateLogitsWithPoint(workDir, clickJSONPath, 1)
}

func (p *Planner) RemoveClickPoint(workDir, clickJSONPath string) (string, error) {
	return updateLogitsWithPoint(workDir, clickJSONPath, 0)
}

func (p *Planner) AddArea(workDir, polygonJSONPath string) (string, error) {
	return updateLogitsWithPolygon(workDir, polygonJSONPath, true)
}

func (p *Planner) RemoveArea(workDir, polygonJSONPath string) (string, error) {
	return updateLogitsWithPolygon(workDir, polygonJSONPath, false)
}

func (p *Planner) ClearCurrent(workDir string) error {
	return clearCurrentFiles(workDir)
}

func postprocessFirstMask(mask utils.Mask, areaMask utils.Mask) utils.Mask {
	cleaned := utils.RemoveSmallComponents(mask, 400)
	filled := utils.FillSmallHoles(cleaned, 400)
	return applyAreaConstraint(filled, areaMask)
}

func applyAreaConstraint(mask utils.Mask, areaMask utils.Mask) utils.Mask {
	if areaMask.Width == 0 || areaMask.Height == 0 {
		return mask
	}
	areaBin := utils.Binarize(areaMask, 1)
	return utils.AndMask(mask, areaBin)
}

func loadAreaMask(workDir string) (utils.Mask, error) {
	path := filepath.Join(workDir, areaMaskFile)
	if !fileExists(path) {
		return utils.Mask{}, nil
	}
	return npy.ReadUint8(path)
}

func loadState(workDir string) (State, error) {
	path := filepath.Join(workDir, stateFile)
	if !fileExists(path) {
		return State{}, fmt.Errorf("缺少 state.json，请先执行初始规划")
	}
	var state State
	if err := readJSON(path, &state); err != nil {
		return State{}, err
	}
	return state, nil
}

func loadClicks(path string, requireLabel bool) ([]Click, error) {
	if !fileExists(path) {
		return []Click{}, nil
	}
	var payload ClickPayload
	if err := readJSON(path, &payload); err == nil && len(payload.Clicks) > 0 {
		if err := validateLabels(path, requireLabel); err != nil {
			return nil, err
		}
		return payload.Clicks, nil
	}

	var raw []Click
	if err := readJSON(path, &raw); err == nil && len(raw) > 0 {
		if err := validateLabels(path, requireLabel); err != nil {
			return nil, err
		}
		return raw, nil
	}
	return []Click{}, nil
}

func toModelClicks(clicks []Click) []utils.Click {
	out := make([]utils.Click, 0, len(clicks))
	for _, c := range clicks {
		out = append(out, utils.Click{X: c.X, Y: c.Y, Label: c.Label})
	}
	return out
}

func updateLogitsWithPoint(workDir, clickJSONPath string, value float32) (string, error) {
	state, err := loadState(workDir)
	if err != nil {
		return "", err
	}
	logitsPath := filepath.Join(workDir, currentLogitsFile)
	if !fileExists(logitsPath) {
		return "", fmt.Errorf("缺少 current_logits.npy，无法修改 logits")
	}
	logits, err := npy.ReadFloat32(logitsPath)
	if err != nil {
		return "", err
	}
	point, err := loadSinglePoint(clickJSONPath)
	if err != nil {
		return "", err
	}
	x, y := scalePoint(point, logits.Width, logits.Height, state)
	applySquare(&logits, x, y, 30, value)
	if err := npy.WriteFloat32(logitsPath, logits); err != nil {
		return "", err
	}
	return logitsPath, nil
}

func updateLogitsWithPolygon(workDir, polygonJSONPath string, add bool) (string, error) {
	state, err := loadState(workDir)
	if err != nil {
		return "", err
	}
	logitsPath := filepath.Join(workDir, currentLogitsFile)
	if !fileExists(logitsPath) {
		return "", fmt.Errorf("缺少 current_logits.npy，无法修改 logits")
	}
	logits, err := npy.ReadFloat32(logitsPath)
	if err != nil {
		return "", err
	}
	polygon, err := loadPolygon(polygonJSONPath)
	if err != nil {
		return "", err
	}
	if len(polygon) < 3 {
		return "", fmt.Errorf("多边形至少需要 3 个点")
	}
	scaled := scalePolygon(polygon, logits.Width, logits.Height, state)
	mask := utils.RasterizePolygon(logits.Width, logits.Height, scaled)
	for i, v := range mask.Data {
		if v == 0 {
			continue
		}
		if add {
			logits.Data[i] = 1
		} else {
			logits.Data[i] = 0
		}
	}
	if err := npy.WriteFloat32(logitsPath, logits); err != nil {
		return "", err
	}
	return logitsPath, nil
}

func applySquare(logits *utils.FloatMask, x, y, size int, value float32) {
	half := size / 2
	x0 := max(0, x-half)
	x1 := min(logits.Width, x+half)
	y0 := max(0, y-half)
	y1 := min(logits.Height, y+half)
	for yy := y0; yy < y1; yy++ {
		for xx := x0; xx < x1; xx++ {
			logits.Set(xx, yy, value)
		}
	}
}

func loadSinglePoint(path string) (Point, error) {
	var raw map[string]any
	if err := readJSON(path, &raw); err != nil {
		return Point{}, err
	}
	if xVal, ok := raw["x"]; ok {
		if yVal, okY := raw["y"]; okY {
			x, okX := toInt(xVal)
			y, okY2 := toInt(yVal)
			if okX && okY2 {
				return Point{X: x, Y: y}, nil
			}
		}
	}
	if pointVal, ok := raw["point"]; ok {
		if pointMap, ok2 := pointVal.(map[string]any); ok2 {
			x, okX := toInt(pointMap["x"])
			y, okY := toInt(pointMap["y"])
			if okX && okY {
				return Point{X: x, Y: y}, nil
			}
		}
	}
	return Point{}, fmt.Errorf("点击点 JSON 需包含 x/y 或 point.x/point.y")
}

func loadPolygon(path string) ([]Point, error) {
	var payload PolygonPayload
	if err := readJSON(path, &payload); err == nil && len(payload.Points) > 0 {
		return payload.Points, nil
	}
	var points []Point
	if err := readJSON(path, &points); err == nil && len(points) > 0 {
		return points, nil
	}
	return nil, fmt.Errorf("多边形 JSON 格式无效")
}

func scalePoint(point Point, width, height int, state State) (int, int) {
	scaleX := float64(width) / float64(state.ImageWidth)
	scaleY := float64(height) / float64(state.ImageHeight)
	return int(float64(point.X) * scaleX), int(float64(point.Y) * scaleY)
}

func scalePolygon(points []Point, width, height int, state State) []image.Point {
	scaleX := float64(width) / float64(state.ImageWidth)
	scaleY := float64(height) / float64(state.ImageHeight)
	out := make([]image.Point, 0, len(points))
	for _, pt := range points {
		out = append(out, image.Point{
			X: int(float64(pt.X) * scaleX),
			Y: int(float64(pt.Y) * scaleY),
		})
	}
	return out
}

func ensureDir(path string) error {
	return os.MkdirAll(path, 0o755)
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
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func absPath(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		return abs
	}
	return path
}

func copyFile(src, dst string) error {
	input, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, input, 0o644)
}

func clearSchemeFiles(workDir string) error {
	entries, err := os.ReadDir(workDir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		name := entry.Name()
		if matchedScheme(name) {
			if err := os.Remove(filepath.Join(workDir, name)); err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}
	return nil
}

func matchedScheme(name string) bool {
	return len(name) > 7 && name[:7] == "scheme_"
}

func listSchemeIndices(workDir string) ([]int, error) {
	entries, err := os.ReadDir(workDir)
	if err != nil {
		return nil, err
	}
	indices := make([]int, 0)
	for _, entry := range entries {
		var idx int
		if _, err := fmt.Sscanf(entry.Name(), "scheme_%d_mask.npy", &idx); err == nil {
			indices = append(indices, idx)
		}
	}
	return indices, nil
}

func clearCurrentFiles(workDir string) error {
	_ = os.Remove(filepath.Join(workDir, currentMaskFile))
	_ = os.Remove(filepath.Join(workDir, currentLogitsFile))
	_ = os.Remove(filepath.Join(workDir, currentClicksFile))
	return nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func validateLabels(path string, requireLabel bool) error {
	if !requireLabel {
		return nil
	}
	var raw any
	if err := readJSON(path, &raw); err != nil {
		return err
	}
	var list []any
	switch typed := raw.(type) {
	case map[string]any:
		data, ok := typed["clicks"]
		if !ok {
			data = typed["points"]
		}
		if data == nil {
			return fmt.Errorf("点击点 JSON 格式无效")
		}
		var okList bool
		list, okList = data.([]any)
		if !okList {
			return fmt.Errorf("点击点 JSON 格式无效")
		}
	case []any:
		list = typed
	default:
		return fmt.Errorf("点击点 JSON 格式无效")
	}
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return fmt.Errorf("点击点应为对象列表")
		}
		if _, ok := obj["label"]; !ok {
			return fmt.Errorf("点击点缺少 label")
		}
	}
	return nil
}

func toInt(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	case int64:
		return int(v), true
	case json.Number:
		if i, err := v.Int64(); err == nil {
			return int(i), true
		}
	}
	return 0, false
}
