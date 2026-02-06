package utils

import "testing"

func TestSelectKthFloat64(t *testing.T) {
	values := []float64{0.4, 2.5, -1.0, 8.2, 3.3, 2.5}
	ordered := []float64{-1.0, 0.4, 2.5, 2.5, 3.3, 8.2}
	for k, expected := range ordered {
		got := SelectKthFloat64(append([]float64(nil), values...), k)
		if got != expected {
			t.Fatalf("k=%d got %v want %v", k, got, expected)
		}
	}
}

func TestSelectKthFloat64ClampIndex(t *testing.T) {
	values := []float64{4.0, 1.0, 9.0}
	if got := SelectKthFloat64(append([]float64(nil), values...), -3); got != 1.0 {
		t.Fatalf("negative index clamp failed: got %v want 1.0", got)
	}
	if got := SelectKthFloat64(append([]float64(nil), values...), 9); got != 9.0 {
		t.Fatalf("overflow index clamp failed: got %v want 9.0", got)
	}
}
