package planner

import (
	"image"
	"image/color"
	"image/draw"
	"math"

	"eyesam/standalone_app_v3_5_go/internal/utils"
)

func ComputeFAZCenter(mask utils.Mask) image.Point {
	if mask.Width == 0 || mask.Height == 0 {
		return image.Point{}
	}
	return image.Point{X: mask.Width / 2, Y: mask.Height / 2}
}

func GenerateRingPoints(center image.Point, radius float64, minDistance int) []image.Point {
	if radius <= 0 || minDistance <= 0 {
		return nil
	}
	if radius*2 < float64(minDistance) {
		return nil
	}
	angleStep := 2 * math.Asin(float64(minDistance)/(2*radius))
	if angleStep <= 0 {
		return nil
	}
	numPoints := int(math.Floor(2 * math.Pi / angleStep))
	if numPoints < 1 {
		return nil
	}
	points := make([]image.Point, 0, numPoints)
	for i := 0; i < numPoints; i++ {
		angle := 2 * math.Pi * float64(i) / float64(numPoints)
		x := center.X + int(math.Round(radius*math.Cos(angle)))
		y := center.Y + int(math.Round(radius*math.Sin(angle)))
		points = append(points, image.Point{X: x, Y: y})
	}
	return points
}

func PlanSurgery(
	img image.Image,
	mask utils.Mask,
	fazCenter image.Point,
	spotDiameter int,
	spotDistance int,
) utils.PlanResult {
	bounds := img.Bounds()
	overlay := image.NewRGBA(bounds)
	draw.Draw(overlay, bounds, img, bounds.Min, draw.Src)

	circleRadius := int(math.Max(1, math.Round(float64(spotDiameter)/2)))
	radiusStep := int(math.Max(1, math.Round(float64(spotDiameter+spotDistance))))
	minDistance := int(math.Max(1, math.Round(float64(spotDiameter+spotDistance))))

	maxRadius := int(math.Max(
		hypot(fazCenter.X, fazCenter.Y),
		math.Max(
			hypot(bounds.Dx()-1-fazCenter.X, fazCenter.Y),
			hypot(fazCenter.X, bounds.Dy()-1-fazCenter.Y),
		),
	))

	var curvePoints [][]image.Point
	var centers []image.Point

	for radius := radiusStep; radius <= maxRadius; radius += radiusStep {
		ringPoints := GenerateRingPoints(fazCenter, float64(radius), minDistance)
		if len(ringPoints) > 0 {
			curvePoints = append(curvePoints, ringPoints)
		}
		for _, pt := range ringPoints {
			if pt.X < 0 || pt.Y < 0 || pt.X >= mask.Width || pt.Y >= mask.Height {
				continue
			}
			if mask.At(pt.X, pt.Y) == 0 {
				continue
			}
			centers = append(centers, pt)
		}
	}

	for _, center := range centers {
		drawCircle(overlay, center, circleRadius, color.RGBA{R: 0, G: 112, B: 255, A: 255})
	}

	return utils.PlanResult{Overlay: overlay, CurvePoints: curvePoints, CircleCenters: centers}
}

func drawCircle(img *image.RGBA, center image.Point, radius int, stroke color.RGBA) {
	if radius <= 0 {
		return
	}
	minX := center.X - radius
	maxX := center.X + radius
	minY := center.Y - radius
	maxY := center.Y + radius
	for y := minY; y <= maxY; y++ {
		for x := minX; x <= maxX; x++ {
			dx := x - center.X
			dy := y - center.Y
			if dx*dx+dy*dy > radius*radius {
				continue
			}
			if x < 0 || y < 0 || x >= img.Bounds().Dx() || y >= img.Bounds().Dy() {
				continue
			}
			img.SetRGBA(x, y, stroke)
		}
	}
}

func hypot(x, y int) float64 {
	return math.Hypot(float64(x), float64(y))
}
