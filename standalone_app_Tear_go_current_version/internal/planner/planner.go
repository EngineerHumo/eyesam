package planner

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"math/rand"

	"eyesam/standalone_app_Tear_go_current_version/internal/utils"
)

func circleMask(width, height int, centers []image.Point, radius int) utils.Mask {
	mask := utils.NewMask(width, height)
	for _, c := range centers {
		for y := c.Y - radius; y <= c.Y+radius; y++ {
			for x := c.X - radius; x <= c.X+radius; x++ {
				dx := x - c.X
				dy := y - c.Y
				if dx*dx+dy*dy <= radius*radius {
					mask.Set(x, y, 1)
				}
			}
		}
	}
	return mask
}

func collectComponentMasks(mask utils.Mask) []utils.Mask {
	num, labels, _ := utils.ConnectedComponents(utils.Binarize(mask, 1), 8)
	components := make([]utils.Mask, 0, max(0, num-1))
	for label := 1; label < num; label++ {
		component := utils.NewMask(mask.Width, mask.Height)
		for idx, v := range labels.Data {
			if int(v) == label {
				component.Data[idx] = 1
			}
		}
		if utils.MaskArea(component) > 0 {
			components = append(components, component)
		}
	}
	return components
}

func contourPoints(mask utils.Mask, spacing float64) []image.Point {
	if spacing < 1 {
		spacing = 1
	}
	border := make([]image.Point, 0)
	for y := 0; y < mask.Height; y++ {
		for x := 0; x < mask.Width; x++ {
			if mask.At(x, y) == 0 {
				continue
			}
			isBorder := false
			for _, d := range []image.Point{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}} {
				if mask.At(x+d.X, y+d.Y) == 0 {
					isBorder = true
					break
				}
			}
			if isBorder {
				border = append(border, image.Point{X: x, Y: y})
			}
		}
	}
	if len(border) == 0 {
		return nil
	}
	samples := make([]image.Point, 0, len(border))
	step := int(math.Max(1, math.Round(spacing)))
	for i := 0; i < len(border); i += step {
		samples = append(samples, border[i])
	}
	return samples
}

func dilate(mask utils.Mask, radius int) utils.Mask {
	if radius <= 0 {
		return mask.Clone()
	}
	out := utils.NewMask(mask.Width, mask.Height)
	r2 := radius * radius
	for y := 0; y < mask.Height; y++ {
		for x := 0; x < mask.Width; x++ {
			if mask.At(x, y) == 0 {
				continue
			}
			for yy := y - radius; yy <= y+radius; yy++ {
				for xx := x - radius; xx <= x+radius; xx++ {
					dx := xx - x
					dy := yy - y
					if dx*dx+dy*dy <= r2 {
						out.Set(xx, yy, 1)
					}
				}
			}
		}
	}
	return out
}

func removeOverlaps(centers []image.Point, minCenterDistance float64) []image.Point {
	if len(centers) < 2 {
		return centers
	}
	remaining := append([]image.Point(nil), centers...)
	minDist2 := minCenterDistance * minCenterDistance
	changed := true
	for changed {
		changed = false
		for i := 0; i < len(remaining); i++ {
			for j := i + 1; j < len(remaining); j++ {
				dx := float64(remaining[i].X - remaining[j].X)
				dy := float64(remaining[i].Y - remaining[j].Y)
				if dx*dx+dy*dy < minDist2 {
					removeIdx := i
					if rand.Intn(2) == 1 {
						removeIdx = j
					}
					remaining = append(remaining[:removeIdx], remaining[removeIdx+1:]...)
					changed = true
					break
				}
			}
			if changed {
				break
			}
		}
	}
	return remaining
}

func pushOutward(point image.Point, center image.Point, existing []image.Point, minCenterDistance float64, bounds image.Rectangle, maxSteps int) image.Point {
	x, y := float64(point.X), float64(point.Y)
	dx := x - float64(center.X)
	dy := y - float64(center.Y)
	norm := math.Hypot(dx, dy)
	if norm == 0 {
		dx, dy, norm = 1, 0, 1
	}
	stepX := dx / norm
	stepY := dy / norm
	minDist2 := minCenterDistance * minCenterDistance
	for step := 0; step < maxSteps; step++ {
		ok := true
		for _, pt := range existing {
			ddx := float64(pt.X) - x
			ddy := float64(pt.Y) - y
			if ddx*ddx+ddy*ddy < minDist2 {
				ok = false
				break
			}
		}
		if ok {
			break
		}
		x += stepX
		y += stepY
		if int(math.Round(x)) < bounds.Min.X || int(math.Round(y)) < bounds.Min.Y || int(math.Round(x)) >= bounds.Max.X || int(math.Round(y)) >= bounds.Max.Y {
			break
		}
	}
	return image.Point{X: int(math.Round(x)), Y: int(math.Round(y))}
}

func PlanSurgery(img image.Image, mask utils.Mask, areaMask *utils.Mask, fazMask *utils.Mask, spotDiameter, spotDistance, maxLayers int) utils.PlanResult {
	bounds := img.Bounds()
	overlay := image.NewRGBA(bounds)
	draw.Draw(overlay, bounds, img, bounds.Min, draw.Src)

	radius := int(math.Max(1, math.Round(float64(spotDiameter)/2)))
	minCenterDistance := float64(spotDiameter + spotDistance)

	maskBin := utils.FillSmallHoles(utils.Binarize(mask, 1), 200)
	var fazBin *utils.Mask
	if fazMask != nil {
		tmp := utils.LargestConnectedComponent(utils.Binarize(*fazMask, 1))
		fazBin = &tmp
	}

	allCenters := make([]image.Point, 0)
	curvePoints := make([][]image.Point, 0)
	for _, component := range collectComponentMasks(maskBin) {
		componentCenters := make([]image.Point, 0)
		curveMask := component.Clone()
		componentCenter := utils.ConnectedComponentCentroid(component)
		for layer := 0; layer < maxLayers; layer++ {
			dilateRadius := radius
			if layer > 0 {
				dilateRadius = radius + spotDistance
			}
			dilated := dilate(curveMask, dilateRadius)
			points := contourPoints(dilated, minCenterDistance*0.85)
			if len(points) > 0 {
				curvePoints = append(curvePoints, points)
			}
			layerAdded := make([]image.Point, 0)
			for _, pt := range points {
				if !pt.In(image.Rect(0, 0, maskBin.Width, maskBin.Height)) {
					continue
				}
				if maskBin.At(pt.X, pt.Y) == 1 {
					continue
				}
				if fazBin != nil && fazBin.At(pt.X, pt.Y) == 1 {
					continue
				}
				adjusted := pushOutward(pt, componentCenter, componentCenters, minCenterDistance, image.Rect(0, 0, maskBin.Width, maskBin.Height), 200)
				if !adjusted.In(image.Rect(0, 0, maskBin.Width, maskBin.Height)) {
					continue
				}
				if fazBin != nil && fazBin.At(adjusted.X, adjusted.Y) == 1 {
					continue
				}
				layerAdded = append(layerAdded, adjusted)
			}
			verified := make([]image.Point, 0, len(layerAdded))
			for _, pt := range layerAdded {
				keep := true
				for _, existing := range verified {
					dx := float64(existing.X - pt.X)
					dy := float64(existing.Y - pt.Y)
					if dx*dx+dy*dy < minCenterDistance*minCenterDistance {
						keep = false
						break
					}
				}
				if keep {
					verified = append(verified, pt)
				}
			}
			if len(verified) > 0 {
				componentCenters = append(componentCenters, verified...)
				curveMask = circleMask(curveMask.Width, curveMask.Height, verified, radius)
			} else {
				curveMask = dilated
			}
		}
		allCenters = append(allCenters, componentCenters...)
	}

	allCenters = removeOverlaps(allCenters, minCenterDistance)
	filteredCenters := make([]image.Point, 0, len(allCenters))
	for _, pt := range allCenters {
		if fazBin != nil && pt.In(image.Rect(0, 0, fazBin.Width, fazBin.Height)) && fazBin.At(pt.X, pt.Y) == 1 {
			continue
		}
		filteredCenters = append(filteredCenters, pt)
		drawCircleOutline(overlay, pt, radius, color.RGBA{B: 255, A: 255}, 2)
	}
	_ = areaMask
	return utils.PlanResult{Overlay: overlay, CurvePoints: curvePoints, CircleCenters: filteredCenters}
}

func drawCircleOutline(img *image.RGBA, center image.Point, radius int, stroke color.RGBA, width int) {
	if radius <= 0 {
		return
	}
	inner := max(0, radius-width)
	outer2 := radius * radius
	inner2 := inner * inner
	for y := center.Y - radius; y <= center.Y+radius; y++ {
		for x := center.X - radius; x <= center.X+radius; x++ {
			if !image.Pt(x, y).In(img.Bounds()) {
				continue
			}
			dx := x - center.X
			dy := y - center.Y
			dist2 := dx*dx + dy*dy
			if dist2 <= outer2 && dist2 >= inner2 {
				img.SetRGBA(x, y, stroke)
			}
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
