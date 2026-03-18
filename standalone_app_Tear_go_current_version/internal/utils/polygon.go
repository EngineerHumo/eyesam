package utils

import (
	"image"
	"sort"
)

func RasterizePolygon(width, height int, points []image.Point) Mask {
	mask := NewMask(width, height)
	if len(points) < 3 || width <= 0 || height <= 0 {
		return mask
	}
	for y := 0; y < height; y++ {
		intersections := make([]int, 0, len(points))
		for i := 0; i < len(points); i++ {
			j := (i + 1) % len(points)
			p1 := points[i]
			p2 := points[j]
			if p1.Y == p2.Y {
				continue
			}
			minY := p1.Y
			maxY := p2.Y
			if minY > maxY {
				minY, maxY = maxY, minY
			}
			if y < minY || y >= maxY {
				continue
			}
			x := float64(p1.X) + float64(y-p1.Y)*float64(p2.X-p1.X)/float64(p2.Y-p1.Y)
			intersections = append(intersections, int(x))
		}
		if len(intersections) < 2 {
			continue
		}
		sort.Ints(intersections)
		for i := 0; i+1 < len(intersections); i += 2 {
			start := intersections[i]
			end := intersections[i+1]
			if start > end {
				start, end = end, start
			}
			if start < 0 {
				start = 0
			}
			if end >= width {
				end = width - 1
			}
			for x := start; x <= end; x++ {
				mask.Set(x, y, 1)
			}
		}
	}
	return mask
}
