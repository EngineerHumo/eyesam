package prp

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"

	"rakrock/app/business/pkg/utils/log"
	"rakrock/app/business/rocktype"
)

var logger = log.Logger.Named("rak_prp")

type PlanAction string

// 2. 定義常量，並明確類型為 PlanAction
const (
	ActionInitialPlan         PlanAction = "InitialPlan"
	ActionSelectInitialScheme PlanAction = "SelectInitialScheme"
	ActionApplyClicks         PlanAction = "ApplyClicks"
	ActionPreviewClicks       PlanAction = "PreviewClicks"
	ActionClearCurrentPlan    PlanAction = "ClearCurrentPlan"
)

func RunPrp(imgPath, outPath string) (any, error) {
	if imgPath == "" || outPath == "" {
		logger.Error("image or output path is empty")
		return "", errors.New("image or output path is empty")
	}

	tmpOnnxPath := filepath.Join("ai-model", "onnx")

	planner, _ := NewSurgicalInterface(tmpOnnxPath)
	result, err := planner.InitialPlan(imgPath, outPath)
	if err != nil {
		logger.Error("Failed to run prp", err)
		return nil, err
	} else {
		logger.Info("Succeed to run prp with create ", "result", result)
	}
	return planner, nil
}

func RunPrpAction(modelIns any, action string, point rocktype.Point, selected *string, workPath string) (any, error) {
	if workPath == "" {
		logger.Error("workPath is empty")
		return "", errors.New("workPath is empty")
	}

	// tmpOnnxPath := filepath.Join("ai-model", "onnx")

	if action != string(ActionSelectInitialScheme) && action != string(ActionApplyClicks) && action != string(ActionPreviewClicks) && action != string(ActionClearCurrentPlan) {
		return nil, errors.New("invalid action")
	}

	planIns, ok := modelIns.(*SurgicalInterface)
	if !ok {
		return nil, errors.New("Faield to get model ")
	}
	if planIns == nil {
		return nil, errors.New("new model is nil")
	}

	switch action {
	case string(ActionSelectInitialScheme):
		err := planIns.SelectInitialScheme(0)
		if err != nil {
			logger.Error("Failed to run prp", err)
			return nil, err
		} else {
			logger.Info("Succeed to run prp with SelectInitialScheme ")
		}
		return nil, nil
	case string(ActionClearCurrentPlan):
		err := planIns.ClearCurrentPlan()
		if err != nil {
			logger.Error("Failed to run prp", err)
			return nil, err
		} else {
			logger.Info("Succeed to run prp with ClearCurrentPlan")
		}
		return nil, nil
	case string(ActionApplyClicks):
		tmpJsonPath := filepath.Join(workPath, "input_click.json")
		err := BuildClickFile(tmpJsonPath, point)
		if err != nil {
			logger.Error("Failed to build clickailed", err)
			return nil, err
		}
		if selected != nil {
			num, err := extractFileNumber(*selected)
			if err != nil {
				logger.Error("Failed to extractFileNumber", err)
				return nil, err
			}
			err = planIns.SelectInitialScheme(num)
			if err != nil {
				logger.Error("Failed to SelectInitialScheme", err)
				return nil, err
			}
		}

		result, err := planIns.ApplyClicks(tmpJsonPath)
		if err != nil {
			logger.Error("Failed to run prp", err)
			return nil, err
		} else {
			logger.Info("Succeed to run prp with ActionApplyClicks ", result)
		}
		return result, nil
	case string(ActionPreviewClicks):
		result, err := planIns.PreviewClicks(workPath)
		if err != nil {
			logger.Error("Failed to run prp", err)
			return nil, err
		} else {
			logger.Info("Succeed to run prp with ActionPreviewClicks ", result)
		}
		return result, nil
	default:
		return nil, errors.New("invalid action")
	}
}

func createFileWithDirs(path string) (*os.File, error) {
	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}
	}
	return os.Create(path)
}

func GetOnnxModelPath() string {
	return filepath.Join("ai-model", "onnx", "first.onnx")
}

func SaveMaskAsPNG(data []float32, width, height int, filename string) {
	// 1. 定义颜色表 (4个类别对应的颜色)
	colors := []color.RGBA{
		{0, 0, 0, 255},   // 类别 0: 黑色 (通常是背景)
		{255, 0, 0, 255}, // 类别 1: 红色
		{0, 255, 0, 255}, // 类别 2: 绿色
		{0, 0, 255, 255}, // 类别 3: 蓝色
	}

	// 2. 创建一个新的 RGBA 图像
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// 3. 填充数据
	// 注意：ONNX 输出格式通常为 [C][H][W]
	stride := width * height
	channels := 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixelIdx := y*width + x

			// 找到 4 个通道中值最大的索引
			maxIdx := 0
			maxVal := data[pixelIdx] // 假设通道 0 是初始最大值
			for c := 1; c < channels+1; c++ {
				val := data[c*stride+pixelIdx]
				if val > maxVal {
					maxVal = val
					maxIdx = c
				}
			}

			// 设置像素颜色
			img.Set(x, y, colors[maxIdx])
		}
	}

	// 4. 保存为文件
	f, err := createFileWithDirs(filename)
	if err != nil {
		fmt.Println("Failed to createFileWithDirs", err)
		return
	}
	defer f.Close()
	png.Encode(f, img)
}

//
// ---------- 通用文件创建 ----------
//

func CreateFileWithDirs(path string) (*os.File, error) {
	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, err
		}
	}
	return os.Create(path)
}

func BuildClickFile(path string, click rocktype.Point) error {
	f, err := CreateFileWithDirs(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// onnx是用0来表示负向，rakai是用-1来表示负向，为了不使用默认值做参数，增强错误检查
	if click.Label == -1 {
		click.Label = 0
	}

	type Click struct {
		Points []rocktype.Point `json:"points"`
	}
	tmpCli := Click{Points: []rocktype.Point{click}}
	if data, err := json.Marshal(tmpCli); err != nil {
		return err
	} else {
		_, err = f.Write(data)
		return err
	}

}

var fileNumRegex = regexp.MustCompile(`mask_(\d+)_rtplan`)

func extractFileNumber(filename string) (int, error) {
	// 查找匹配项
	matches := fileNumRegex.FindStringSubmatch(filename)

	// matches[0] 是完整匹配字符串，matches[1] 是第一个括号捕获的数字分组
	if len(matches) < 2 {
		return 0, fmt.Errorf("无法在文件名 '%s' 中找到数字编号", filename)
	}

	// 将提取的字符串转换为整数
	num, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, fmt.Errorf("数字转换失败: %v", err)
	}

	return num, nil
}

//
// ---------- Mask -> NPY ----------
//

// SaveMaskNPY
// 保存 float32 HxW mask，兼容 numpy.load
func SaveMaskNPY(mask []float32, outPath string, h, w int) error {
	if len(mask) < h*w {
		return fmt.Errorf("mask len %d < %d", len(mask), h*w)
	}

	f, err := CreateFileWithDirs(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(buildNPYHeader(h, w)); err != nil {
		return err
	}

	buf := make([]byte, 4*h*w)
	for i := 0; i < h*w; i++ {
		binary.LittleEndian.PutUint32(
			buf[i*4:(i+1)*4],
			math.Float32bits(mask[i]),
		)
	}

	_, err = f.Write(buf)
	return err
}

func buildNPYHeader(h, w int) []byte {
	magic := []byte("\x93NUMPY")
	version := []byte{0x01, 0x00}

	headerDict := "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
		intToStr(h) + ", " + intToStr(w) + ",), }"

	headerLen := len(headerDict) + 1
	pad := (16 - ((10 + headerLen) % 16)) % 16
	for i := 0; i < pad; i++ {
		headerDict += " "
	}
	headerDict += "\n"

	headerSize := make([]byte, 2)
	binary.LittleEndian.PutUint16(headerSize, uint16(len(headerDict)))

	out := make([]byte, 0, 10+len(headerDict))
	out = append(out, magic...)
	out = append(out, version...)
	out = append(out, headerSize...)
	out = append(out, []byte(headerDict)...)

	return out
}

// func buildNPYHeader(h, w int) []byte {
// 	// magic + version
// 	magic := []byte("\x93NUMPY")
// 	version := []byte{0x01, 0x00}

// 	headerDict := "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
// 		intToStr(h) + ", " + intToStr(w) + "), }"

// 	// padding to 16-byte alignment
// 	headerLen := len(headerDict) + 1
// 	pad := (16 - ((10 + headerLen) % 16)) % 16
// 	for i := 0; i < pad; i++ {
// 		headerDict += " "
// 	}
// 	headerDict += "\n"

// 	headerSize := make([]byte, 2)
// 	binary.LittleEndian.PutUint16(headerSize, uint16(len(headerDict)))

// 	out := make([]byte, 0, 10+len(headerDict))
// 	out = append(out, magic...)
// 	out = append(out, version...)
// 	out = append(out, headerSize...)
// 	out = append(out, []byte(headerDict)...)

// 	return out
// }

func intToStr(v int) string {
	if v == 0 {
		return "0"
	}
	buf := make([]byte, 0)
	for v > 0 {
		buf = append([]byte{byte('0' + v%10)}, buf...)
		v /= 10
	}
	return string(buf)
}

func SaveMaskToPng(
	data []float32,
	shape []int64,
	outPath string,
	threshold float32,
) error {

	if len(shape) != 4 {
		return fmt.Errorf("expect [B,C,H,W], got %v", shape)
	}

	h := int(shape[2])
	w := int(shape[3])

	expected := h * w
	if len(data) < expected {
		return fmt.Errorf("data len %d < %d", len(data), expected)
	}

	f, err := CreateFileWithDirs(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	img := image.NewGray(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			v := data[idx]

			if threshold >= 0 {
				if v >= threshold {
					img.SetGray(x, y, color.Gray{Y: 255})
				} else {
					img.SetGray(x, y, color.Gray{Y: 0})
				}
			} else {
				if v < 0 {
					v = 0
				}
				if v > 1 {
					v = 1
				}
				img.SetGray(x, y, color.Gray{Y: uint8(v * 255)})
			}
		}
	}

	err = png.Encode(f, img)
	if err != nil {
		fmt.Println("failed to encode png:", err)
		return err
	}
	return nil
}
