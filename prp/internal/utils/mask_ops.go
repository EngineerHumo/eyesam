package utils

import (
	"container/heap"
	"fmt"
	"image"
	"log"
	"math"
)

func NormalizeImageBytes(data []uint8) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(v) / 255.0
	}
	return out
}

func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func SigmoidMask(mask FloatMask) FloatMask {
	out := NewFloatMask(mask.Width, mask.Height)
	for i, v := range mask.Data {
		out.Data[i] = Sigmoid(v)
	}
	return out
}

func ResizeMaskNearest(mask Mask, width, height int) Mask {
	if width <= 0 || height <= 0 || mask.Width == 0 || mask.Height == 0 {
		return mask
	}
	out := NewMask(width, height)
	for y := 0; y < height; y++ {
		sy := int(float64(y) * float64(mask.Height) / float64(height))
		for x := 0; x < width; x++ {
			sx := int(float64(x) * float64(mask.Width) / float64(width))
			out.Set(x, y, mask.At(sx, sy))
		}
	}
	return out
}

func ResizeFloatMaskLinear(mask FloatMask, width, height int) FloatMask {
	if width <= 0 || height <= 0 || mask.Width == 0 || mask.Height == 0 {
		return mask
	}
	out := NewFloatMask(width, height)
	scaleX := float64(mask.Width-1) / float64(max(1, width-1))
	scaleY := float64(mask.Height-1) / float64(max(1, height-1))
	for y := 0; y < height; y++ {
		fy := float64(y) * scaleY
		y0 := int(math.Floor(fy))
		y1 := min(y0+1, mask.Height-1)
		wy := float32(fy - float64(y0))
		for x := 0; x < width; x++ {
			fx := float64(x) * scaleX
			x0 := int(math.Floor(fx))
			x1 := min(x0+1, mask.Width-1)
			wx := float32(fx - float64(x0))
			v00 := mask.At(x0, y0)
			v01 := mask.At(x1, y0)
			v10 := mask.At(x0, y1)
			v11 := mask.At(x1, y1)
			top := v00*(1-wx) + v01*wx
			bottom := v10*(1-wx) + v11*wx
			out.Set(x, y, top*(1-wy)+bottom*wy)
		}
	}
	return out
}

func FillSmallHoles(mask Mask, areaThreshold int) Mask {
	if mask.Width == 0 || mask.Height == 0 {
		return mask
	}
	binary := Binarize(mask, 1)
	inverted := NewMask(mask.Width, mask.Height)
	for i, v := range binary.Data {
		if v == 0 {
			inverted.Data[i] = 1
		}
	}
	num, labels, stats := ConnectedComponents(inverted, 8)
	filled := binary.Clone()
	for label := 1; label < num; label++ {
		st := stats[label]
		if st.Area >= areaThreshold {
			continue
		}
		if st.MinX == 0 || st.MinY == 0 || st.MaxX == mask.Width-1 || st.MaxY == mask.Height-1 {
			continue
		}
		for i, v := range labels.Data {
			if v == uint8(label) {
				filled.Data[i] = 1
			}
		}
	}
	return filled
}

func RemoveSmallComponents(mask Mask, minSize int) Mask {
	if mask.Width == 0 || mask.Height == 0 {
		return mask
	}
	binary := Binarize(mask, 1)
	num, labels, stats := ConnectedComponents(binary, 8)
	if num <= 1 {
		return binary
	}
	filtered := binary.Clone()
	for label := 1; label < num; label++ {
		if stats[label].Area < minSize {
			for i, v := range labels.Data {
				if v == uint8(label) {
					filtered.Data[i] = 0
				}
			}
		}
	}
	return filtered
}

func LargestConnectedComponent(mask Mask) Mask {
	if mask.Width == 0 || mask.Height == 0 {
		return mask
	}
	binary := Binarize(mask, 1)
	num, labels, stats := ConnectedComponents(binary, 8)
	if num <= 1 {
		return binary
	}
	largestLabel := 1
	largestArea := stats[1].Area
	for label := 2; label < num; label++ {
		if stats[label].Area > largestArea {
			largestArea = stats[label].Area
			largestLabel = label
		}
	}
	out := NewMask(mask.Width, mask.Height)
	for i, v := range labels.Data {
		if v == uint8(largestLabel) {
			out.Data[i] = 1
		}
	}
	return out
}

func MaskArea(mask Mask) int {
	area := 0
	for _, v := range mask.Data {
		if v > 0 {
			area++
		}
	}
	return area
}

func ConnectedComponentCentroid(mask Mask) image.Point {
	if mask.Width == 0 || mask.Height == 0 {
		return image.Point{}
	}
	maxValue := uint8(0)
	for _, v := range mask.Data {
		if v > maxValue {
			maxValue = v
			break
		}
	}
	if maxValue == 0 {
		log.Printf("Mask is empty, fallback to image center")
		return image.Point{X: mask.Width / 2, Y: mask.Height / 2}
	}
	var sumX, sumY, count int
	for idx, v := range mask.Data {
		if v == 0 {
			continue
		}
		x := idx % mask.Width
		y := idx / mask.Width
		sumX += x
		sumY += y
		count++
	}
	if count == 0 {
		return image.Point{X: mask.Width / 2, Y: mask.Height / 2}
	}
	x := int(math.Round(float64(sumX) / float64(count)))
	y := int(math.Round(float64(sumY) / float64(count)))
	return image.Point{X: x, Y: y}
}

type ComponentStats struct {
	Area int
	MinX int
	MinY int
	MaxX int
	MaxY int
}

func ConnectedComponents(mask Mask, connectivity int) (int, Mask, []ComponentStats) {
	if connectivity != 4 && connectivity != 8 {
		connectivity = 8
	}
	labels := NewMask(mask.Width, mask.Height)
	stats := []ComponentStats{{Area: 0, MinX: mask.Width, MinY: mask.Height, MaxX: -1, MaxY: -1}}
	current := 0
	dirs := []image.Point{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	if connectivity == 8 {
		dirs = append(dirs, image.Point{1, 1}, image.Point{-1, 1}, image.Point{1, -1}, image.Point{-1, -1})
	}

	for y := 0; y < mask.Height; y++ {
		for x := 0; x < mask.Width; x++ {
			if mask.At(x, y) == 0 || labels.At(x, y) != 0 {
				continue
			}
			current++
			stats = append(stats, ComponentStats{Area: 0, MinX: x, MinY: y, MaxX: x, MaxY: y})
			queue := []image.Point{{X: x, Y: y}}
			labels.Set(x, y, uint8(current))
			for len(queue) > 0 {
				p := queue[0]
				queue = queue[1:]
				st := &stats[current]
				st.Area++
				if p.X < st.MinX {
					st.MinX = p.X
				}
				if p.Y < st.MinY {
					st.MinY = p.Y
				}
				if p.X > st.MaxX {
					st.MaxX = p.X
				}
				if p.Y > st.MaxY {
					st.MaxY = p.Y
				}
				for _, d := range dirs {
					nx := p.X + d.X
					ny := p.Y + d.Y
					if nx < 0 || ny < 0 || nx >= mask.Width || ny >= mask.Height {
						continue
					}
					if mask.At(nx, ny) == 0 || labels.At(nx, ny) != 0 {
						continue
					}
					labels.Set(nx, ny, uint8(current))
					queue = append(queue, image.Point{X: nx, Y: ny})
				}
			}
		}
	}
	return current + 1, labels, stats
}

type node struct {
	idx    int
	dist   float32
	source int
}

type priorityQueue []node

func (pq priorityQueue) Len() int           { return len(pq) }
func (pq priorityQueue) Less(i, j int) bool { return pq[i].dist < pq[j].dist }
func (pq priorityQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *priorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(node))
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[:n-1]
	return item
}

func distanceTransform(mask Mask) ([]float32, []int) {
	w := mask.Width
	h := mask.Height
	if w == 0 || h == 0 {
		return nil, nil
	}
	dist := make([]float32, w*h)
	labels := make([]int, w*h)
	for i := range dist {
		dist[i] = float32(math.Inf(1))
		labels[i] = -1
	}
	pq := priorityQueue{}
	zeroCoords := make([]int, 0)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if mask.At(x, y) == 0 {
				dist[idx] = 0
				labels[idx] = len(zeroCoords)
				zeroCoords = append(zeroCoords, idx)
				heap.Push(&pq, node{idx: idx, dist: 0, source: labels[idx]})
			}
		}
	}
	neighbors := []struct {
		dx   int
		dy   int
		cost float32
	}{{1, 0, 1}, {-1, 0, 1}, {0, 1, 1}, {0, -1, 1}, {1, 1, float32(math.Sqrt2)}, {-1, 1, float32(math.Sqrt2)}, {1, -1, float32(math.Sqrt2)}, {-1, -1, float32(math.Sqrt2)}}
	for pq.Len() > 0 {
		cur := heap.Pop(&pq).(node)
		if cur.dist > dist[cur.idx] {
			continue
		}
		x := cur.idx % w
		y := cur.idx / w
		for _, n := range neighbors {
			nx := x + n.dx
			ny := y + n.dy
			if nx < 0 || ny < 0 || nx >= w || ny >= h {
				continue
			}
			nidx := ny*w + nx
			nd := cur.dist + n.cost
			if nd < dist[nidx] {
				dist[nidx] = nd
				labels[nidx] = cur.source
				heap.Push(&pq, node{idx: nidx, dist: nd, source: cur.source})
			}
		}
	}
	return dist, labels
}

func DistanceTransform(mask Mask) []float32 {
	dist, _ := distanceTransform(mask)
	return dist
}

func FillRemovedRegions(labels Mask, removed Mask, targetValue uint8) Mask {
	if labels.Width == 0 || labels.Height == 0 {
		return labels
	}
	binary := NewMask(labels.Width, labels.Height)
	allOnes := true
	for i, v := range labels.Data {
		if v == targetValue {
			binary.Data[i] = 1
		} else {
			allOnes = false
		}
	}
	if allOnes {
		out := labels.Clone()
		for i, v := range removed.Data {
			if v != 0 {
				out.Data[i] = targetValue
			}
		}
		return out
	}
	dist, labelIdx := distanceTransform(binary)
	if dist == nil {
		return labels
	}
	zeroCoords := make([]int, 0)
	for i, v := range binary.Data {
		if v == 0 {
			zeroCoords = append(zeroCoords, i)
		}
	}
	out := labels.Clone()
	for i, v := range removed.Data {
		if v == 0 {
			continue
		}
		idx := labelIdx[i]
		if idx < 0 || idx >= len(zeroCoords) {
			continue
		}
		coordIdx := zeroCoords[idx]
		out.Data[i] = labels.Data[coordIdx]
	}
	_ = dist
	return out
}

func GaussianBlur(mask Mask, sigma float64) []float64 {
	width := mask.Width
	height := mask.Height
	if width == 0 || height == 0 {
		return nil
	}
	kernelRadius := int(math.Ceil(3 * sigma))
	kernelSize := kernelRadius*2 + 1
	kernel := make([]float64, kernelSize)
	var sum float64
	for i := -kernelRadius; i <= kernelRadius; i++ {
		value := math.Exp(-(float64(i * i)) / (2 * sigma * sigma))
		kernel[i+kernelRadius] = value
		sum += value
	}
	for i := range kernel {
		kernel[i] /= sum
	}
	temp := make([]float64, width*height)
	out := make([]float64, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var accum float64
			for k := -kernelRadius; k <= kernelRadius; k++ {
				xx := clamp(x+k, 0, width-1)
				idx := y*width + xx
				accum += float64(mask.Data[idx]) * kernel[k+kernelRadius]
			}
			temp[y*width+x] = accum
		}
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var accum float64
			for k := -kernelRadius; k <= kernelRadius; k++ {
				yy := clamp(y+k, 0, height-1)
				idx := yy*width + x
				accum += temp[idx] * kernel[k+kernelRadius]
			}
			out[y*width+x] = accum
		}
	}
	return out
}

func MajorityFilterOnBand(labels Mask, band Mask, kernelSize int) Mask {
	if labels.Width == 0 || labels.Height == 0 {
		return labels
	}
	counts := make([][]int, 4)
	for i := 0; i < 4; i++ {
		counts[i] = make([]int, labels.Width*labels.Height)
	}
	radius := kernelSize / 2
	for y := 0; y < labels.Height; y++ {
		for x := 0; x < labels.Width; x++ {
			idx := y*labels.Width + x
			for yy := y - radius; yy <= y+radius; yy++ {
				if yy < 0 || yy >= labels.Height {
					continue
				}
				for xx := x - radius; xx <= x+radius; xx++ {
					if xx < 0 || xx >= labels.Width {
						continue
					}
					v := labels.At(xx, yy)
					if v < 4 {
						counts[v][idx]++
					}
				}
			}
		}
	}
	out := labels.Clone()
	for i, v := range band.Data {
		if v == 0 {
			continue
		}
		maxLabel := 0
		maxCount := counts[0][i]
		for label := 1; label < 4; label++ {
			if counts[label][i] > maxCount {
				maxCount = counts[label][i]
				maxLabel = label
			}
		}
		out.Data[i] = uint8(maxLabel)
	}
	return out
}

func BoundaryBand(labels Mask, kernelSize int) Mask {
	out := NewMask(labels.Width, labels.Height)
	for value := uint8(0); value < 4; value++ {
		mask := EqualMask(labels, value)
		dilated := Dilate(mask, kernelSize, false)
		eroded := Erode(mask, kernelSize, false)
		for i := range out.Data {
			if dilated.Data[i] != eroded.Data[i] {
				out.Data[i] = 1
			}
		}
	}
	return out
}

func DilateRed(labels Mask) Mask {
	red := EqualMask(labels, 3)
	if MaskArea(red) == 0 {
		return labels
	}
	dilated := Dilate(red, 3, false)
	out := labels.Clone()
	for i, v := range dilated.Data {
		if v != 0 {
			out.Data[i] = 3
		}
	}
	return out
}

func ReplaceComponentWithNeighbors(labels Mask, component Mask, value uint8) Mask {
	if MaskArea(component) == 0 {
		return labels
	}
	border := Dilate(component, 3, false)
	for i := range border.Data {
		if component.Data[i] != 0 {
			border.Data[i] = 0
		}
	}
	if MaskArea(border) == 0 {
		out := labels.Clone()
		for i, v := range component.Data {
			if v != 0 {
				out.Data[i] = 0
			}
		}
		return out
	}
	counts := make([]int, 4)
	for i, v := range border.Data {
		if v == 0 {
			continue
		}
		label := labels.Data[i]
		if label < 4 {
			counts[label]++
		}
	}
	counts[value] = 0
	newValue := uint8(0)
	maxCount := counts[0]
	for i := 1; i < len(counts); i++ {
		if counts[i] > maxCount {
			maxCount = counts[i]
			newValue = uint8(i)
		}
	}
	out := labels.Clone()
	for i, v := range component.Data {
		if v != 0 {
			out.Data[i] = newValue
		}
	}
	return out
}

func CleanGreenComponents(labels Mask) Mask {
	green := EqualMask(labels, 1)
	total := MaskArea(green)
	if total == 0 {
		return labels
	}
	threshold := max(total/10, 1)
	num, comp, stats := ConnectedComponents(green, 8)
	out := labels.Clone()
	for idx := 1; idx < num; idx++ {
		if stats[idx].Area < threshold {
			component := NewMask(labels.Width, labels.Height)
			for i, v := range comp.Data {
				if v == uint8(idx) {
					component.Data[i] = 1
				}
			}
			out = ReplaceComponentWithNeighbors(out, component, 1)
		}
	}
	return out
}

func KeepLargestComponent(labels Mask, value uint8) Mask {
	mask := EqualMask(labels, value)
	if MaskArea(mask) == 0 {
		return labels
	}
	num, comp, stats := ConnectedComponents(mask, 8)
	if num <= 1 {
		return labels
	}
	largest := 1
	maxArea := stats[1].Area
	for i := 2; i < num; i++ {
		if stats[i].Area > maxArea {
			maxArea = stats[i].Area
			largest = i
		}
	}
	out := labels.Clone()
	for idx := 1; idx < num; idx++ {
		if idx == largest {
			continue
		}
		component := NewMask(labels.Width, labels.Height)
		for i, v := range comp.Data {
			if v == uint8(idx) {
				component.Data[i] = 1
			}
		}
		out = ReplaceComponentWithNeighbors(out, component, value)
	}
	return out
}

func OpeningAndRefill(labels Mask, value uint8, radius int) Mask {
	mask := EqualMask(labels, value)
	if MaskArea(mask) == 0 {
		return labels
	}
	kernel := EllipseKernel(radius)
	opened := MorphOpen(mask, kernel)
	removed := NewMask(mask.Width, mask.Height)
	for i, v := range mask.Data {
		if v != 0 && opened.Data[i] == 0 {
			removed.Data[i] = 1
		}
	}
	out := labels.Clone()
	for i, v := range mask.Data {
		if v != 0 {
			out.Data[i] = value
		}
	}
	return FillRemovedRegions(out, removed, value)
}

func AreaPreservingRethreshold(labels Mask) Mask {
	originalBlack := EqualMask(labels, 0)
	originalRed := EqualMask(labels, 3)
	yellow := EqualMask(labels, 2)
	green := EqualMask(labels, 1)
	yellowGreen := OrMask(yellow, green)
	if MaskArea(yellowGreen) == 0 {
		return labels
	}
	protected := Dilate(OrMask(originalBlack, originalRed), 7, false)
	movable := AndMask(yellowGreen, NotMask(protected))
	out := labels.Clone()
	if MaskArea(movable) == 0 {
		applyMaskValue(&out, originalBlack, 0)
		applyMaskValue(&out, originalRed, 3)
		return out
	}
	bandKernel := EllipseKernel(25)
	dilatedY := DilateWithKernel(yellow, bandKernel)
	dilatedG := DilateWithKernel(green, bandKernel)
	band := AndMask(movable, AndMask(dilatedY, dilatedG))
	if MaskArea(band) == 0 {
		band = movable
	}
	yellowFixed := AndMask(yellow, NotMask(band))
	yellowTarget := MaskArea(yellow)
	yellowToAllocate := yellowTarget - MaskArea(yellowFixed)
	if yellowToAllocate <= 0 {
		applyMaskValue(&out, band, 1)
		applyMaskValue(&out, yellowFixed, 2)
		applyMaskValue(&out, originalBlack, 0)
		applyMaskValue(&out, originalRed, 3)
		return out
	}
	bandSize := MaskArea(band)
	if yellowToAllocate >= bandSize {
		applyMaskValue(&out, band, 2)
		applyMaskValue(&out, yellowFixed, 2)
		applyMaskValue(&out, originalBlack, 0)
		applyMaskValue(&out, originalRed, 3)
		return out
	}
	blur := GaussianBlur(yellow, 12.0)
	values := make([]float64, 0, bandSize)
	indices := make([]int, 0, bandSize)
	for i, v := range band.Data {
		if v != 0 {
			values = append(values, blur[i])
			indices = append(indices, i)
		}
	}
	if len(values) == 0 {
		return out
	}
	thresholdIndex := len(values) - yellowToAllocate
	copyValues := append([]float64(nil), values...)

	thresholdValue := SelectKthFloat64(copyValues, thresholdIndex)

	newYellow := NewMask(labels.Width, labels.Height)
	selectedCount := 0
	for i, v := range values {
		if v > thresholdValue {
			newYellow.Data[indices[i]] = 1
			selectedCount++
		}
	}
	need := yellowToAllocate - selectedCount
	if need > 0 {
		for i, v := range values {
			if v == thresholdValue {
				newYellow.Data[indices[i]] = 1
				need--
				if need == 0 {
					break
				}
			}
		}
	}
	applyMaskValue(&out, band, 1)
	applyMaskValue(&out, newYellow, 2)
	applyMaskValue(&out, yellowFixed, 2)
	applyMaskValue(&out, originalBlack, 0)
	applyMaskValue(&out, originalRed, 3)
	return out
}

func CleanYellowComponents(labels Mask) Mask {
	yellow := EqualMask(labels, 2)
	total := MaskArea(yellow)
	if total == 0 {
		return labels
	}
	threshold := max(total/10, 1)
	num, comp, stats := ConnectedComponents(yellow, 8)
	if num <= 1 {
		return labels
	}
	out := labels.Clone()
	for idx := 1; idx < num; idx++ {
		if stats[idx].Area < threshold {
			component := NewMask(labels.Width, labels.Height)
			for i, v := range comp.Data {
				if v == uint8(idx) {
					component.Data[i] = 1
				}
			}
			out = ReplaceComponentWithNeighbors(out, component, 2)
		}
	}
	return out
}

func RemoveSmallComponentsByValue(labels Mask, minSize int) Mask {
	out := labels.Clone()
	for value := uint8(0); value < 4; value++ {
		mask := EqualMask(out, value)
		if MaskArea(mask) == 0 {
			continue
		}
		num, comp, stats := ConnectedComponents(mask, 8)
		if num <= 1 {
			continue
		}
		for idx := 1; idx < num; idx++ {
			if stats[idx].Area < minSize {
				component := NewMask(out.Width, out.Height)
				for i, v := range comp.Data {
					if v == uint8(idx) {
						component.Data[i] = 1
					}
				}
				out = ReplaceComponentWithNeighbors(out, component, value)
			}
		}
	}
	return out
}

func ProcessLabels(labels Mask) Mask {
	band := BoundaryBand(labels, 5)
	labels = MajorityFilterOnBand(labels, band, 5)
	labels = DilateRed(labels)
	labels = CleanGreenComponents(labels)
	labels = KeepLargestComponent(labels, 3)
	labels = KeepLargestComponent(labels, 0)
	labels = OpeningAndRefill(labels, 1, 4)
	labels = OpeningAndRefill(labels, 2, 4)
	labels = AreaPreservingRethreshold(labels)
	labels = CleanYellowComponents(labels)
	labels = RemoveSmallComponentsByValue(labels, 1000)
	return labels
}

func EqualMask(labels Mask, value uint8) Mask {
	out := NewMask(labels.Width, labels.Height)
	for i, v := range labels.Data {
		if v == value {
			out.Data[i] = 1
		}
	}
	return out
}

func NotMask(mask Mask) Mask {
	out := NewMask(mask.Width, mask.Height)
	for i, v := range mask.Data {
		if v == 0 {
			out.Data[i] = 1
		}
	}
	return out
}

func OrMask(a, b Mask) Mask {
	out := NewMask(a.Width, a.Height)
	for i := range out.Data {
		if a.Data[i] != 0 || b.Data[i] != 0 {
			out.Data[i] = 1
		}
	}
	return out
}

func AndMask(a, b Mask) Mask {
	out := NewMask(a.Width, a.Height)
	for i := range out.Data {
		if a.Data[i] != 0 && b.Data[i] != 0 {
			out.Data[i] = 1
		}
	}
	return out
}

func applyMaskValue(labels *Mask, mask Mask, value uint8) {
	if labels == nil {
		return
	}
	for i, v := range mask.Data {
		if v != 0 {
			labels.Data[i] = value
		}
	}
}

func Dilate(mask Mask, kernelSize int, ellipse bool) Mask {
	if kernelSize <= 1 {
		return mask
	}
	kernel := SquareKernel(kernelSize)
	if ellipse {
		kernel = EllipseKernel(kernelSize / 2)
	}
	return DilateWithKernel(mask, kernel)
}

func Erode(mask Mask, kernelSize int, ellipse bool) Mask {
	if kernelSize <= 1 {
		return mask
	}
	kernel := SquareKernel(kernelSize)
	if ellipse {
		kernel = EllipseKernel(kernelSize / 2)
	}
	return ErodeWithKernel(mask, kernel)
}
func DilateWithKernel(mask Mask, kernel []image.Point) Mask {
	out := NewMask(mask.Width, mask.Height)
	for y := 0; y < mask.Height; y++ {
		base := y * mask.Width
		for x := 0; x < mask.Width; x++ {
			if mask.Data[base+x] == 0 {
				continue
			}
			for _, k := range kernel {
				nx := x + k.X
				ny := y + k.Y
				if nx < 0 || ny < 0 || nx >= mask.Width || ny >= mask.Height {
					continue
				}
				out.Data[ny*mask.Width+nx] = 1
			}
		}
	}
	return out
}

func ErodeWithKernel(mask Mask, kernel []image.Point) Mask {
	out := NewMask(mask.Width, mask.Height)
	for y := 0; y < mask.Height; y++ {
		base := y * mask.Width
		for x := 0; x < mask.Width; x++ {
			if mask.Data[base+x] == 0 {
				continue
			}
			all := true
			for _, k := range kernel {
				nx := x + k.X
				ny := y + k.Y
				if nx < 0 || ny < 0 || nx >= mask.Width || ny >= mask.Height {
					all = false
					break
				}
				if mask.Data[ny*mask.Width+nx] == 0 {
					all = false
					break
				}
			}
			if all {
				out.Data[base+x] = 1
			}
		}
	}
	return out
}

func SelectKthFloat64(values []float64, k int) float64 {
	if len(values) == 0 {
		return 0
	}
	if k < 0 {
		k = 0
	}
	if k >= len(values) {
		k = len(values) - 1
	}
	left := 0
	right := len(values) - 1
	for left < right {
		pivot := partitionFloat64(values, left, right, (left+right)/2)
		switch {
		case k == pivot:
			return values[pivot]
		case k < pivot:
			right = pivot - 1
		default:
			left = pivot + 1
		}
	}
	return values[left]
}

func partitionFloat64(values []float64, left, right, pivot int) int {
	pivotValue := values[pivot]
	values[pivot], values[right] = values[right], values[pivot]
	store := left
	for i := left; i < right; i++ {
		if values[i] < pivotValue {
			values[store], values[i] = values[i], values[store]
			store++
		}
	}
	values[right], values[store] = values[store], values[right]
	return store
}

func MorphOpen(mask Mask, kernel []image.Point) Mask {
	eroded := ErodeWithKernel(mask, kernel)
	return DilateWithKernel(eroded, kernel)
}

func SquareKernel(size int) []image.Point {
	radius := size / 2
	points := make([]image.Point, 0, size*size)
	for y := -radius; y <= radius; y++ {
		for x := -radius; x <= radius; x++ {
			points = append(points, image.Point{X: x, Y: y})
		}
	}
	return points
}

func EllipseKernel(radius int) []image.Point {
	points := make([]image.Point, 0)
	if radius <= 0 {
		points = append(points, image.Point{X: 0, Y: 0})
		return points
	}
	r2 := radius * radius
	for y := -radius; y <= radius; y++ {
		for x := -radius; x <= radius; x++ {
			if x*x+y*y <= r2 {
				points = append(points, image.Point{X: x, Y: y})
			}
		}
	}
	return points
}

func clamp(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
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

func ValidateMaskShape(mask Mask, width, height int) error {
	if mask.Width != width || mask.Height != height {
		return fmt.Errorf("mask shape mismatch: got %dx%d expected %dx%d", mask.Width, mask.Height, width, height)
	}
	return nil
}
