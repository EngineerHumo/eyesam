package inference

import (
	"fmt"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"rakrock/app/business/pkg/utils/log"
	"rakrock/app/business/rakai/prp/internal/onnxutil"
	"rakrock/app/business/rakai/prp/internal/utils"
)

var logger = log.Logger.Named("inference")

type Result struct {
	Mask   utils.Mask
	Logits utils.FloatMask
}

type OnnxModel struct {
	ModelPath       string
	Session         *ort.DynamicAdvancedSession
	Inputs          []ort.InputOutputInfo
	Outputs         []ort.InputOutputInfo
	InputShapes     map[string][]int64
	ImageInputNames []string
}

func NewOnnxModel(path string) *OnnxModel {
	return &OnnxModel{ModelPath: path}
}
func (m *OnnxModel) Init() error {
	return m.initSession()
}

func (m *OnnxModel) initSession() error {
	if m.Session != nil {
		return nil
	}
	if err := onnxutil.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize onnx environment (model=%s): %w", m.ModelPath, err)
	}
	inputs, outputs, err := ort.GetInputOutputInfo(m.ModelPath)
	if err != nil {
		return fmt.Errorf("inspect onnx io (model=%s): %w", m.ModelPath, err)
	}
	m.Inputs = inputs
	m.Outputs = outputs
	m.InputShapes = make(map[string][]int64)
	for _, info := range inputs {
		m.InputShapes[info.Name] = info.Dimensions
	}
	m.ImageInputNames = m.findImageInputNames()
	inputNames := make([]string, len(inputs))
	for i, info := range inputs {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(outputs))
	for i, info := range outputs {
		outputNames[i] = info.Name
	}
	opts, err := onnxutil.SessionOptions()
	if err != nil {
		return fmt.Errorf("build session options (model=%s): %w", m.ModelPath, err)
	}
	defer opts.Destroy()
	session, err := ort.NewDynamicAdvancedSession(m.ModelPath, inputNames, outputNames, opts)
	if err != nil {
		return fmt.Errorf("create onnx session (model=%s): %w", m.ModelPath, err)
	}
	m.Session = session
	return nil
}

func (m *OnnxModel) ImageInputSize(fallbackWidth, fallbackHeight int) (int, int, error) {
	if err := m.initSession(); err != nil {
		return 0, 0, fmt.Errorf("ImageInputSize initSession failed (model=%s): %w", m.ModelPath, err)
	}
	for key, shape := range m.InputShapes {
		if len(shape) != 4 {
			continue
		}
		if key != "image" {
			continue
		}

		if shape[1] == 1 || shape[1] == 3 {
			if shape[2] > 0 && shape[3] > 0 {
				return int(shape[3]), int(shape[2]), nil
			}
		}
		if shape[3] == 1 || shape[3] == 3 {
			if shape[1] > 0 && shape[2] > 0 {
				return int(shape[2]), int(shape[1]), nil
			}
		}
	}
	return 0, 0, fmt.Errorf("ImageInputSize could not determine model input size (model=%s, fallback=%dx%d, shapes=%v)", m.ModelPath, fallbackWidth, fallbackHeight, m.InputShapes)
}

func (m *OnnxModel) EnsureInitialized() error {
	return m.initSession()
}

func (m *OnnxModel) Infer(image utils.ModelImage, clicks []utils.Click, maskInput *utils.FloatMask) (Result, error) {
	if err := m.initSession(); err != nil {
		return Result{}, err
	}
	resizedBounds := image.Resized.Bounds()
	origBounds := image.Original.Bounds()
	resizedHW := [2]int{resizedBounds.Dy(), resizedBounds.Dx()}
	origHW := [2]int{origBounds.Dy(), origBounds.Dx()}
	feed := make(map[string]ort.Value)
	imgInputs, err := m.resolveImageInput(image)
	if err != nil {
		return Result{}, err
	}
	for k, v := range imgInputs {
		logger.Info("onnx input", k, v.GetShape())
		feed[k] = v
	}
	for k, v := range m.resolvePointsInputs(clicks, resizedHW, origHW) {
		logger.Info("onnx input", k, v.GetShape())
		feed[k] = v
	}
	for k, v := range m.resolveMaskInputs(maskInput, resizedHW) {
		logger.Info("onnx input", k, v.GetShape())
		feed[k] = v
	}
	for k, v := range m.resolveOrigSizeInputs(origHW) {
		logger.Info("onnx input", k, v.GetShape())
		feed[k] = v
	}
	logger.Info("Infer function onnx input schema:", m.InputShapes)
	inputs := make([]ort.Value, len(m.Inputs))
	for i, info := range m.Inputs {
		val, ok := feed[info.Name]
		if !ok {
			return Result{}, fmt.Errorf("missing input %s", info.Name)
		}
		logger.Info("Infer fill input", info.Name, val.GetShape())
		inputs[i] = val
	}
	outputs := make([]ort.Value, len(m.Outputs))
	if err := m.Session.Run(inputs, outputs); err != nil {
		m.destroyValues(feed)
		return Result{}, fmt.Errorf("run onnx model: %w", err)
	}
	m.destroyValues(feed)
	defer destroyValues(outputs)
	if len(outputs) == 0 {
		return Result{}, fmt.Errorf("onnx model returned no outputs")
	}
	logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return Result{}, fmt.Errorf("unexpected output tensor type")
	}
	logits, mask := tensorToMasks(logitsTensor)
	return Result{Mask: mask, Logits: logits}, nil
}

func (m *OnnxModel) InputShape(name string) ([]int64, bool) {
	shape, ok := m.InputShapes[name]
	return shape, ok
}

func (m *OnnxModel) resolveImageInput(image utils.ModelImage) (map[string]ort.Value, error) {
	if len(m.ImageInputNames) == 0 {
		return nil, fmt.Errorf("no valid image input found in ONNX model")
	}
	name := m.ImageInputNames[0]
	shape := m.InputShapes[name]
	img := utils.NormalizeImageBytes(utils.ImageToRGBBytes(image.Resized))
	logger.Info("resolveImageInput shape:", fmt.Sprintf("%+v len: %d", shape, len(img)))
	channelsFirst := len(shape) == 4 && (shape[1] == 1 || shape[1] == 3)
	channelIdx := int(shape[1])
	var data []float32
	if channelsFirst {
		data = make([]float32, len(img))
		idx := 0
		for c := 0; c < channelIdx; c++ {
			for i := c; i < len(img); i += 3 {
				data[idx] = img[i]
				idx++
			}
		}
	} else {
		data = img
	}
	resizedBounds := image.Resized.Bounds()
	shapeDims := ort.Shape{1, int64(resizedBounds.Dy()), int64(resizedBounds.Dx()), 3}
	if channelsFirst {
		shapeDims = ort.Shape{1, 3, int64(resizedBounds.Dy()), int64(resizedBounds.Dx())}
	}
	inputTensor, err := ort.NewTensor(shapeDims, data)
	if err != nil {
		return nil, fmt.Errorf("create image tensor: %w", err)
	}
	return map[string]ort.Value{name: inputTensor}, nil
}

func scaleClicksFromOriginal(clicks []utils.Click, resizedHW, origHW [2]int) (float32, float32) {
	if origHW[1] <= 0 || origHW[0] <= 0 || resizedHW[1] <= 0 || resizedHW[0] <= 0 {
		return 1, 1
	}
	return float32(resizedHW[1]) / float32(origHW[1]), float32(resizedHW[0]) / float32(origHW[0])
}

func (m *OnnxModel) resolvePointsInputs(clicks []utils.Click, resizedHW, origHW [2]int) map[string]ort.Value {
	if len(clicks) == 0 {
		return map[string]ort.Value{}
	}
	points := make([]float32, 0, len(clicks)*2)
	labels := make([]int64, 0, len(clicks))
	scaleX, scaleY := scaleClicksFromOriginal(clicks, resizedHW, origHW)
	logger.Info("prompt:", "clicks len", len(clicks), "scale=", fmt.Sprintf("(%.4f,%.4f)", scaleX, scaleY))
	for _, c := range clicks {
		points = append(points, float32(c.X)*scaleX, float32(c.Y)*scaleY)
		labels = append(labels, int64(c.Label))
	}

	for _, v := range points {
		logger.Info("input point value:", " ", fmt.Sprintf("(%.4f)", v))
	}

	pointsShape := ort.Shape{1, int64(len(clicks)), 2}
	labelsShape := ort.Shape{1, int64(len(clicks))}
	pointsTensor, err := ort.NewTensor(pointsShape, points)
	if err != nil {
		logger.Info("create points tensor: %v", err)
		return map[string]ort.Value{}
	}
	labelsTensor, err := ort.NewTensor(labelsShape, labels)
	if err != nil {
		logger.Info("create labels tensor: %v", err)
		_ = pointsTensor.Destroy()
		return map[string]ort.Value{}
	}
	inputs := make(map[string]ort.Value)
	for name, shape := range m.InputShapes {
		if len(shape) == 3 && shape[len(shape)-1] == 2 {
			inputs[name] = pointsTensor
		}
		if strings.Contains(name, "point_labels") || strings.Contains(name, "label") {
			if _, exists := inputs[name]; !exists {
				inputs[name] = labelsTensor
			}
		}
		if strings.Contains(name, "point_coords") {
			inputs[name] = pointsTensor
		}
	}
	return inputs
}

func (m *OnnxModel) resolveMaskInputs(maskInput *utils.FloatMask, resizedHW [2]int) map[string]ort.Value {
	inputs := make(map[string]ort.Value)
	for name, shape := range m.InputShapes {
		if strings.Contains(name, "has_mask") {
			value := float32(0)
			if maskInput != nil {
				value = 1
			}
			shapeDims := fillDynamicShape(shape)
			if len(shapeDims) == 0 {
				shapeDims = []int64{1}
			}
			data := make([]float32, shapeElements(shapeDims))
			for i := range data {
				data[i] = value
			}
			val, err := ort.NewTensor(ort.Shape(shapeDims), data)
			if err == nil {
				inputs[name] = val
			}
		}
	}
	for name, shape := range m.InputShapes {
		if strings.Contains(name, "has_mask") {
			continue
		}
		if strings.Contains(name, "mask_input") || strings.Contains(name, "mask_inputs") {
			if maskInput != nil {
				shapeDims := ort.Shape{1, 1, int64(maskInput.Height), int64(maskInput.Width)}
				val, err := ort.NewTensor(shapeDims, maskInput.Data)
				if err == nil {
					inputs[name] = val
				}
				continue
			}
			shapeDims := fillDynamicShape(shape)
			if len(shapeDims) == 0 {
				shapeDims = []int64{1, 1, int64(resizedHW[0] / 4), int64(resizedHW[1] / 4)}
			}
			if len(shapeDims) == 4 && (shapeDims[2] <= 0 || shapeDims[3] <= 0) {
				shapeDims[2] = int64(resizedHW[0] / 4)
				shapeDims[3] = int64(resizedHW[1] / 4)
			}
			data := make([]float32, shapeElements(shapeDims))
			val, err := ort.NewTensor(ort.Shape(shapeDims), data)
			if err == nil {
				inputs[name] = val
			}
		}
	}
	return inputs
}

func (m *OnnxModel) resolveOrigSizeInputs(origHW [2]int) map[string]ort.Value {
	inputs := make(map[string]ort.Value)
	for name, shape := range m.InputShapes {
		if !strings.Contains(name, "orig") {
			continue
		}
		if len(shape) == 2 && shape[1] == 2 {
			data := []float32{float32(origHW[0]), float32(origHW[1])}
			val, err := ort.NewTensor(ort.Shape{1, 2}, data)
			if err == nil {
				inputs[name] = val
			}
		} else if len(shape) == 1 && shape[0] == 2 {
			data := []float32{float32(origHW[0]), float32(origHW[1])}
			val, err := ort.NewTensor(ort.Shape{2}, data)
			if err == nil {
				inputs[name] = val
			}
		}
	}
	return inputs
}

func tensorToMasks(tensor *ort.Tensor[float32]) (utils.FloatMask, utils.Mask) {
	shape := tensor.GetShape()
	data := tensor.GetData()
	var height, width int
	if len(shape) == 4 {
		height = int(shape[2])
		width = int(shape[3])
		data = data[:height*width]
	} else if len(shape) == 3 {
		height = int(shape[1])
		width = int(shape[2])
		data = data[:height*width]
	}
	logits := utils.NewFloatMask(width, height)
	copy(logits.Data, data)
	maxVal, minVal := float32(-1e9), float32(1e9)
	for _, v := range logits.Data {
		if v > maxVal {
			maxVal = v
		}
		if v < minVal {
			minVal = v
		}
	}
	maskSource := logits
	if maxVal > 1 || minVal < 0 {
		maskSource = utils.SigmoidMask(logits)
	}
	mask := utils.BinarizeFloat(maskSource, 0.5)
	return logits, mask
}

func (m *OnnxModel) findImageInputNames() []string {
	candidates := []string{}
	for name, shape := range m.InputShapes {
		if len(shape) != 4 {
			continue
		}
		if strings.Contains(name, "mask") {
			continue
		}
		channelsFirst := shape[1] == 1 || shape[1] == 3
		channelsLast := shape[3] == 1 || shape[3] == 3
		if channelsFirst || channelsLast {
			candidates = append(candidates, name)
		}
	}
	return candidates
}

func fillDynamicShape(shape []int64) []int64 {
	if len(shape) == 0 {
		return nil
	}
	out := make([]int64, len(shape))
	for i, dim := range shape {
		if dim <= 0 {
			out[i] = 1
		} else {
			out[i] = dim
		}
	}
	return out
}

func shapeElements(shape []int64) int {
	count := int64(1)
	for _, v := range shape {
		if v <= 0 {
			v = 1
		}
		count *= v
	}
	return int(count)
}

func (m *OnnxModel) destroyValues(values map[string]ort.Value) {
	seen := make(map[ort.Value]struct{}, len(values))
	for _, v := range values {
		if v == nil {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		if err := v.Destroy(); err != nil {
			logger.Info("failed to destroy onnx value: %v", err)
		}
	}
}

func destroyValues(values []ort.Value) {
	for _, v := range values {
		if v == nil {
			continue
		}
		if err := v.Destroy(); err != nil {
			logger.Info("failed to destroy onnx value: %v", err)
		}
	}
}
