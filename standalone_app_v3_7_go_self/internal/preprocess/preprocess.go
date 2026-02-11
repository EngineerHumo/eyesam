package preprocess

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"log"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	"eyesam/standalone_app_v3_7_go/internal/onnxutil"
	"eyesam/standalone_app_v3_7_go/internal/utils"
)

var channelToLabel = []uint8{0, 3, 2, 1}

type Result struct {
	Labels   utils.Mask
	FAZMask  utils.Mask
	AreaMask utils.Mask
}

type PreSegmenter struct {
	ModelPath string
	Session   *ort.DynamicAdvancedSession
	Inputs    []ort.InputOutputInfo
	Outputs   []ort.InputOutputInfo
	InputSize [2]int
}

func NewPreSegmenter(modelPath string) *PreSegmenter {
	return &PreSegmenter{ModelPath: modelPath, InputSize: [2]int{1024, 1024}}
}

func (p *PreSegmenter) initSession() error {
	if p.Session != nil {
		return nil
	}
	if err := onnxutil.InitializeEnvironment(); err != nil {
		return err
	}
	inputs, outputs, err := ort.GetInputOutputInfo(p.ModelPath)
	if err != nil {
		return fmt.Errorf("inspect onnx io: %w", err)
	}
	p.Inputs = inputs
	p.Outputs = outputs
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
		return err
	}
	defer opts.Destroy()
	session, err := ort.NewDynamicAdvancedSession(p.ModelPath, inputNames, outputNames, opts)
	if err != nil {
		return fmt.Errorf("create onnx session: %w", err)
	}
	p.Session = session
	return nil
}

func (p *PreSegmenter) Infer(image utils.ModelImage) (Result, error) {
	if err := p.initSession(); err != nil {
		return Result{}, err
	}
	modelInput, pads, processedShape, err := prepareModelInput(image.Original, p.InputSize)
	if err != nil {
		return Result{}, err
	}
	inputTensor, err := ort.NewTensor(ort.Shape{1, 3, int64(p.InputSize[0]), int64(p.InputSize[1])}, modelInput)
	if err != nil {
		return Result{}, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()
	outputs := make([]ort.Value, len(p.Outputs))
	if err := p.Session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return Result{}, fmt.Errorf("run onnx pre model: %w", err)
	}
	defer destroyValues(outputs)
	if len(outputs) == 0 {
		return Result{}, fmt.Errorf("onnx pre model returned no outputs")
	}
	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return Result{}, fmt.Errorf("unexpected output tensor type")
	}
	labels, err := inferLabels(outputTensor, pads, processedShape)
	if err != nil {
		return Result{}, err
	}
	origBounds := image.Original.Bounds()
	if processedShape[0] != origBounds.Dy() || processedShape[1] != origBounds.Dx() {
		labels = utils.ResizeMaskNearest(labels, origBounds.Dx(), origBounds.Dy())
	}
	labels = utils.ProcessLabels(labels)
	fazMask := utils.EqualMask(labels, 3)
	areaMask := utils.EqualMask(labels, 1)
	return Result{Labels: labels, FAZMask: fazMask, AreaMask: areaMask}, nil
}

func prepareModelInput(img image.Image, inputSize [2]int) ([]float32, [4]int, [2]int, error) {
	origBounds := img.Bounds()
	origH := origBounds.Dy()
	origW := origBounds.Dx()
	targetH, targetW := inputSize[0], inputSize[1]
	resized := img
	if origH > targetH || origW > targetW {
		resized = utils.ResizeBilinear(img, targetW, targetH)
	}
	processedShape := [2]int{resized.Bounds().Dy(), resized.Bounds().Dx()}
	padded, pads, err := padToShapeImage(resized, targetH, targetW)
	if err != nil {
		return nil, [4]int{}, [2]int{}, err
	}
	data := utils.NormalizeImageBytes(utils.ImageToRGBBytes(padded))
	input := make([]float32, targetH*targetW*3)
	idx := 0
	for c := 0; c < 3; c++ {
		for i := c; i < len(data); i += 3 {
			input[idx] = data[i]
			idx++
		}
	}
	return input, pads, processedShape, nil
}

func padToShapeImage(img image.Image, targetH, targetW int) (image.Image, [4]int, error) {
	bounds := img.Bounds()
	h := bounds.Dy()
	w := bounds.Dx()
	if h > targetH || w > targetW {
		return nil, [4]int{}, fmt.Errorf("image size %dx%d exceeds target %dx%d", h, w, targetH, targetW)
	}
	padTop := (targetH - h) / 2
	padBottom := targetH - h - padTop
	padLeft := (targetW - w) / 2
	padRight := targetW - w - padLeft
	canvas := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	draw.Draw(canvas, canvas.Bounds(), &image.Uniform{C: color.Black}, image.Point{}, draw.Src)
	draw.Draw(canvas, image.Rect(padLeft, padTop, padLeft+w, padTop+h), img, bounds.Min, draw.Src)
	return canvas, [4]int{padTop, padBottom, padLeft, padRight}, nil
}

func inferLabels(output *ort.Tensor[float32], pads [4]int, outputShape [2]int) (utils.Mask, error) {
	data := output.GetData()
	shape := output.GetShape()
	if len(shape) != 4 {
		return utils.Mask{}, fmt.Errorf("onnx output shape mismatch, expected (N,C,H,W) got %v", shape)
	}
	channels := int(shape[1])
	height := int(shape[2])
	width := int(shape[3])
	if channels < 1 {
		return utils.Mask{}, fmt.Errorf("onnx output has no channels")
	}
	padTop, padBottom, padLeft, padRight := pads[0], pads[1], pads[2], pads[3]
	startH := padTop
	endH := height - padBottom
	startW := padLeft
	endW := width - padRight
	cropH := endH - startH
	cropW := endW - startW
	if cropH != outputShape[0] || cropW != outputShape[1] {
		return utils.Mask{}, fmt.Errorf("cropped shape mismatch: %dx%d vs %dx%d", cropH, cropW, outputShape[0], outputShape[1])
	}
	labels := utils.NewMask(cropW, cropH)
	for y := 0; y < cropH; y++ {
		for x := 0; x < cropW; x++ {
			maxIdx := 0
			maxVal := float32(math.Inf(-1))
			for c := 0; c < channels; c++ {
				idx := ((0*channels+c)*height+(y+startH))*width + (x + startW)
				val := data[idx]
				if val > maxVal {
					maxVal = val
					maxIdx = c
				}
			}
			label := uint8(0)
			if maxIdx < len(channelToLabel) {
				label = channelToLabel[maxIdx]
			}
			labels.Set(x, y, label)
		}
	}
	return labels, nil
}

func destroyValues(values []ort.Value) {
	for _, v := range values {
		if v == nil {
			continue
		}
		if err := v.Destroy(); err != nil {
			log.Printf("failed to destroy onnx value: %v", err)
		}
	}
}
