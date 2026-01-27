package utils

import "image"

type ModelImage struct {
	Original image.Image
	Resized  image.Image
	ScaleX   float64
	ScaleY   float64
}

type Click struct {
	X     float64
	Y     float64
	Label int
}

type PlanResult struct {
	Overlay       image.Image
	CurvePoints   [][]image.Point
	CircleCenters []image.Point
}

type Mask struct {
	Width  int
	Height int
	Data   []uint8
}

type FloatMask struct {
	Width  int
	Height int
	Data   []float32
}

const (
	CircleRadius        = 12
	DefaultSpotDiameter = 25
	DefaultSpotDistance = 25
	MinSpotDiameter     = 10
	MaxSpotDiameter     = 100
	MinSpotDistance     = 3
	MaxSpotDistance     = 30
)

func NewMask(width, height int) Mask {
	return Mask{Width: width, Height: height, Data: make([]uint8, width*height)}
}

func NewFloatMask(width, height int) FloatMask {
	return FloatMask{Width: width, Height: height, Data: make([]float32, width*height)}
}

func (m Mask) Index(x, y int) int {
	return y*m.Width + x
}

func (m Mask) At(x, y int) uint8 {
	if x < 0 || y < 0 || x >= m.Width || y >= m.Height {
		return 0
	}
	return m.Data[m.Index(x, y)]
}

func (m Mask) Set(x, y int, v uint8) {
	if x < 0 || y < 0 || x >= m.Width || y >= m.Height {
		return
	}
	m.Data[m.Index(x, y)] = v
}

func (m Mask) Clone() Mask {
	dup := make([]uint8, len(m.Data))
	copy(dup, m.Data)
	return Mask{Width: m.Width, Height: m.Height, Data: dup}
}

func (m FloatMask) Index(x, y int) int {
	return y*m.Width + x
}

func (m FloatMask) At(x, y int) float32 {
	if x < 0 || y < 0 || x >= m.Width || y >= m.Height {
		return 0
	}
	return m.Data[m.Index(x, y)]
}

func (m FloatMask) Set(x, y int, v float32) {
	if x < 0 || y < 0 || x >= m.Width || y >= m.Height {
		return
	}
	m.Data[m.Index(x, y)] = v
}

func (m FloatMask) Clone() FloatMask {
	dup := make([]float32, len(m.Data))
	copy(dup, m.Data)
	return FloatMask{Width: m.Width, Height: m.Height, Data: dup}
}

func Binarize(mask Mask, threshold uint8) Mask {
	out := NewMask(mask.Width, mask.Height)
	for i, v := range mask.Data {
		if v >= threshold {
			out.Data[i] = 1
		}
	}
	return out
}

func BinarizeFloat(mask FloatMask, threshold float32) Mask {
	out := NewMask(mask.Width, mask.Height)
	for i, v := range mask.Data {
		if v >= threshold {
			out.Data[i] = 1
		}
	}
	return out
}
