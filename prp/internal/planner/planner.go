package planner

import (
	"image"

	"rakrock/app/business/rakai/prp/internal/utils"
)

func ComputeFAZCenter(mask utils.Mask) image.Point {
	fazBin := utils.Binarize(mask, 1)
	fazLcc := utils.LargestConnectedComponent(fazBin)
	return utils.InscribedCenter(fazLcc)
}
