package planner

import (
	"image"

	"eyesam/standalone_app_v3_6_go/internal/utils"
)

func ComputeFAZCenter(mask utils.Mask) image.Point {
	fazBin := utils.Binarize(mask, 1)
	fazLcc := utils.LargestConnectedComponent(fazBin)
	return utils.InscribedCenter(fazLcc)
}
