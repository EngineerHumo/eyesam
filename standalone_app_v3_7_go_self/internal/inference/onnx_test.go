package inference

import (
	"math"
	"testing"

	"eyesam/standalone_app_v3_7_go/internal/utils"
)

func TestScaleClicksFromOriginal(t *testing.T) {
	sx, sy := scaleClicksFromOriginal(
		[]utils.Click{{X: 620, Y: 620, Label: 1}},
		[2]int{512, 512},
		[2]int{1240, 1240},
	)

	if math.Abs(float64(sx)-float64(512.0/1240.0)) > 1e-6 {
		t.Fatalf("unexpected scaleX: %f", sx)
	}
	if math.Abs(float64(sy)-float64(512.0/1240.0)) > 1e-6 {
		t.Fatalf("unexpected scaleY: %f", sy)
	}
}

func TestScaleClicksFromOriginalInvalidShapeFallback(t *testing.T) {
	sx, sy := scaleClicksFromOriginal(
		[]utils.Click{{X: 10, Y: 20, Label: 1}},
		[2]int{512, 512},
		[2]int{0, 1240},
	)

	if sx != 1 || sy != 1 {
		t.Fatalf("expected fallback scale=(1,1), got (%f,%f)", sx, sy)
	}
}
