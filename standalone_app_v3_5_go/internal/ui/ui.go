package ui

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"sort"
	"sync"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/driver/desktop"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/widget"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"eyesam/standalone_app_v3_5_go/internal/inference"
	"eyesam/standalone_app_v3_5_go/internal/pipeline"
	"eyesam/standalone_app_v3_5_go/internal/planner"
	"eyesam/standalone_app_v3_5_go/internal/utils"
)

type SchemeData struct {
	Mask    utils.Mask
	Plan    utils.PlanResult
	Circles []image.Point
	Color   color.RGBA
	Logits  *utils.FloatMask
	Click   *utils.Click
}

type AppState struct {
	HasPlan       bool
	CurrentMask   *utils.Mask
	CurrentLogits *utils.FloatMask
	Clicks        []utils.Click
	AutoClick     *utils.Click
	Mode          string
}

type ellipseInfo struct {
	Center image.Point
	Width  float64
	Height float64
	Angle  float64
}

type UI struct {
	app      fyne.App
	window   fyne.Window
	pipeline *pipeline.Pipeline

	state         AppState
	currentImage  *utils.ModelImage
	originalImage image.Image
	displaySize   int
	displayScaleX float64
	displayScaleY float64

	imageView *canvas.Image
	overlay   *imageOverlay

	plan          *utils.PlanResult
	fazCenter     *image.Point
	lastAutoClick *image.Point
	areaMask      *utils.Mask
	fazMask       *utils.Mask
	fazEllipse    *ellipseInfo

	schemes             []SchemeData
	schemeButtons       []*widget.Button
	selectedSchemeIndex *int
	autoMode            bool

	spotDiameter int
	spotDistance int
	exposureTime int

	spotDiameterSlider *widget.Slider
	spotDistanceSlider *widget.Slider
	exposureTimeSlider *widget.Slider

	btnPositive        *widget.Button
	btnNegative        *widget.Button
	btnAddPoint        *widget.Button
	btnRemovePoint     *widget.Button
	btnAddArea         *widget.Button
	btnRemoveArea      *widget.Button
	btnClear           *widget.Button
	btnConfirm         *widget.Button
	btnStartFromRegion *widget.Button

	schemeContainer *fyne.Container

	mouseOver     bool
	lastMousePos  *image.Point
	drawingPoints []image.Point
	previewMutex  sync.Mutex
}

func Run(pipe *pipeline.Pipeline) error {
	ui := &UI{
		app:          app.New(),
		pipeline:     pipe,
		displaySize:  640,
		spotDiameter: utils.DefaultSpotDiameter,
		spotDistance: utils.DefaultSpotDistance,
		exposureTime: 0,
	}
	ui.window = ui.app.NewWindow("手术方案规划工具")
	ui.window.Resize(fyne.NewSize(1100, 800))

	ui.imageView = canvas.NewImageFromImage(image.NewRGBA(image.Rect(0, 0, ui.displaySize, ui.displaySize)))
	ui.imageView.FillMode = canvas.ImageFillContain
	ui.imageView.SetMinSize(fyne.NewSize(float32(ui.displaySize), float32(ui.displaySize)))
	ui.overlay = newImageOverlay(ui)

	imageContainer := container.NewMax(ui.imageView, ui.overlay)
	imageScroll := container.NewScroll(imageContainer)

	ui.buildControls()

	mainContent := container.NewHBox(imageScroll, ui.buildSidePanel())
	ui.window.SetContent(mainContent)

	ui.window.SetMainMenu(ui.buildMenu())
	ui.window.ShowAndRun()
	return nil
}

func (ui *UI) buildMenu() *fyne.MainMenu {
	openItem := fyne.NewMenuItem("打开", func() {
		dlg := dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
			if err != nil {
				dialog.ShowError(err, ui.window)
				return
			}
			if reader == nil {
				return
			}
			path := reader.URI().Path()
			_ = reader.Close()
			ui.openImage(path)
		}, ui.window)
		dlg.SetFilter(storage.NewExtensionFileFilter([]string{".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}))
		dlg.Show()
	})
	fileMenu := fyne.NewMenu("文件", openItem)
	return fyne.NewMainMenu(fileMenu)
}

func (ui *UI) buildControls() {
	ui.btnPositive = widget.NewButton("正向点击点", func() { ui.toggleMode("add_positive") })
	ui.btnNegative = widget.NewButton("负向点击点", func() { ui.toggleMode("add_negative") })
	ui.btnAddPoint = widget.NewButton("添加激光点", func() { ui.toggleMode("add_point") })
	ui.btnRemovePoint = widget.NewButton("删除激光点", func() { ui.toggleMode("remove_point") })
	ui.btnAddArea = widget.NewButton("添加手术区域", func() { ui.toggleMode("add_area") })
	ui.btnRemoveArea = widget.NewButton("删除手术区域", func() { ui.toggleMode("remove_area") })
	ui.btnClear = widget.NewButton("清空当前手术方案", ui.clearPlan)
	ui.btnConfirm = widget.NewButton("确定手术方案", ui.confirmPlan)
	ui.btnStartFromRegion = widget.NewButton("从此区域开始规划", ui.startFromRegion)
	ui.btnStartFromRegion.Disable()

	ui.spotDiameterSlider = widget.NewSlider(float64(utils.MinSpotDiameter), float64(utils.MaxSpotDiameter))
	ui.spotDiameterSlider.Value = float64(ui.spotDiameter)
	ui.spotDiameterSlider.Orientation = widget.Vertical
	ui.spotDiameterSlider.OnChanged = func(v float64) { ui.onSpotSettingChange(int(v), ui.spotDistance, ui.exposureTime) }

	ui.spotDistanceSlider = widget.NewSlider(float64(utils.MinSpotDistance), float64(utils.MaxSpotDistance))
	ui.spotDistanceSlider.Value = float64(ui.spotDistance)
	ui.spotDistanceSlider.Orientation = widget.Vertical
	ui.spotDistanceSlider.OnChanged = func(v float64) { ui.onSpotSettingChange(ui.spotDiameter, int(v), ui.exposureTime) }

	ui.exposureTimeSlider = widget.NewSlider(0, 100)
	ui.exposureTimeSlider.Value = float64(ui.exposureTime)
	ui.exposureTimeSlider.Orientation = widget.Vertical
	ui.exposureTimeSlider.OnChanged = func(v float64) { ui.onSpotSettingChange(ui.spotDiameter, ui.spotDistance, int(v)) }

	ui.btnPositive.Enable()
	ui.updateButtonStates()
	ui.disableSchemeControls(true)
}

func (ui *UI) buildSidePanel() fyne.CanvasObject {
	aiTools := container.NewVBox(
		widget.NewLabelWithStyle("AI工具", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		ui.btnPositive,
		ui.btnNegative,
		widget.NewLabel("建议先使用AI工具再用传统工具修改手术规划"),
	)

	traditionalTools := container.NewVBox(
		widget.NewLabelWithStyle("传统工具", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		ui.btnAddPoint,
		ui.btnRemovePoint,
		ui.btnAddArea,
		ui.btnRemoveArea,
	)

	actionTools := container.NewVBox(ui.btnClear, ui.btnConfirm)

	schemeLabel := widget.NewLabelWithStyle("区域列表", fyne.TextAlignLeading, fyne.TextStyle{Bold: true})
	ui.schemeContainer = container.NewVBox()
	schemePanel := container.NewVBox(schemeLabel, ui.schemeContainer, ui.btnStartFromRegion)

	sliderPanel := container.NewVBox(
		widget.NewLabelWithStyle("参数设置", fyne.TextAlignLeading, fyne.TextStyle{Bold: true}),
		ui.buildSlider("光斑直径", ui.spotDiameterSlider),
		ui.buildSlider("光斑距离", ui.spotDistanceSlider),
		ui.buildSlider("曝光时间", ui.exposureTimeSlider),
	)

	left := container.NewVBox(aiTools, traditionalTools, actionTools)
	side := container.NewHBox(left, schemePanel, sliderPanel)
	return side
}

func (ui *UI) buildSlider(label string, slider *widget.Slider) fyne.CanvasObject {
	return container.NewVBox(widget.NewLabel(label), slider)
}

func (ui *UI) toggleMode(mode string) {
	if mode == "add_negative" && !ui.state.HasPlan {
		dialog.ShowInformation("提示", "请先生成手术方案", ui.window)
		return
	}
	if ui.state.Mode == mode {
		ui.state.Mode = "none"
	} else {
		ui.state.Mode = mode
	}
	ui.refreshToggleStates()
}

func (ui *UI) refreshToggleStates() {
	ui.setButtonActive(ui.btnPositive, ui.state.Mode == "add_positive")
	ui.setButtonActive(ui.btnNegative, ui.state.Mode == "add_negative")
	ui.setButtonActive(ui.btnAddPoint, ui.state.Mode == "add_point")
	ui.setButtonActive(ui.btnRemovePoint, ui.state.Mode == "remove_point")
	ui.setButtonActive(ui.btnAddArea, ui.state.Mode == "add_area")
	ui.setButtonActive(ui.btnRemoveArea, ui.state.Mode == "remove_area")
}

func (ui *UI) setButtonActive(button *widget.Button, active bool) {
	if active {
		button.Importance = widget.HighImportance
	} else {
		button.Importance = widget.MediumImportance
	}
	button.Refresh()
}

func (ui *UI) openImage(path string) {
	img, err := utils.LoadImage(path)
	if err != nil {
		dialog.ShowError(err, ui.window)
		return
	}
	ui.originalImage = img
	modelW, modelH := ui.pipeline.IterationModel.ImageInputSize(img.Bounds().Dx(), img.Bounds().Dy())
	current := utils.PrepareImageForModel(img, modelW, modelH)
	ui.currentImage = &current
	ui.displayScaleX = float64(img.Bounds().Dx()) / float64(ui.displaySize)
	ui.displayScaleY = float64(img.Bounds().Dy()) / float64(ui.displaySize)

	progress := dialog.NewProgress("AI规划中", "处理中...", ui.window)
	progress.Show()
	progressChan := make(chan int, 16)
	resultChan := make(chan struct {
		result pipeline.Result
		err    error
	}, 1)

	go func() {
		res, err := ui.pipeline.RunInitial(img, func(count int) {
			progressChan <- count
		})
		resultChan <- struct {
			result pipeline.Result
			err    error
		}{result: res, err: err}
	}()

	go func() {
		for {
			select {
			case val := <-progressChan:
				ui.app.Driver().RunOnMain(func() {
					progress.SetValue(math.Min(100, float64(val*7)))
				})
			case res := <-resultChan:
				ui.app.Driver().RunOnMain(func() {
					progress.SetValue(100)
					progress.Hide()
					if res.err != nil {
						dialog.ShowError(res.err, ui.window)
						return
					}
					ui.applyInitialResult(res.result)
				})
				return
			}
		}
	}()
}

func (ui *UI) applyInitialResult(result pipeline.Result) {
	ui.spotDiameter = utils.DefaultSpotDiameter
	ui.spotDistance = utils.DefaultSpotDistance
	ui.spotDiameterSlider.Value = float64(ui.spotDiameter)
	ui.spotDistanceSlider.Value = float64(ui.spotDistance)
	ui.exposureTimeSlider.Value = float64(ui.exposureTime)

	if len(result.SchemeMasks) > 0 {
		ui.state.CurrentMask = &result.SchemeMasks[0]
	}
	ui.state.CurrentLogits = &result.CurrentLogits
	ui.state.AutoClick = &result.LastAutoClick
	if result.LastAutoClick.Label != 0 {
		ui.state.Clicks = []utils.Click{result.LastAutoClick}
	} else {
		ui.state.Clicks = nil
	}
	ui.state.HasPlan = true
	ui.state.Mode = "none"
	ui.refreshToggleStates()
	ui.updateButtonStates()

	ui.schemes = nil
	ui.autoMode = false
	ui.selectedSchemeIndex = nil
	ui.refreshSchemeButtons()

	ui.fazCenter = &result.FAZCenter
	ui.lastAutoClick = &result.CurrentClick
	ui.areaMask = &result.AreaMask
	ui.fazMask = &result.FAZMask
	ui.computeFazEllipse()
	log.Printf("initial_plan_clicks=%d", len(ui.state.Clicks))

	if len(result.SchemeMasks) > 0 {
		ui.buildSchemes(result.SchemeMasks, result.SchemeLogits, result.SchemeClicks)
	} else {
		ui.disableSchemeControls(true)
		ui.renderOverlay(ui.originalImage)
	}
}

func (ui *UI) updateButtonStates() {
	if ui.state.HasPlan {
		ui.btnNegative.Enable()
		ui.btnConfirm.Enable()
		ui.btnAddPoint.Enable()
		ui.btnRemovePoint.Enable()
		ui.btnAddArea.Enable()
		ui.btnRemoveArea.Enable()
	} else {
		ui.btnNegative.Disable()
		ui.btnConfirm.Disable()
		ui.btnAddPoint.Disable()
		ui.btnRemovePoint.Disable()
		ui.btnAddArea.Disable()
		ui.btnRemoveArea.Disable()
	}
}

func (ui *UI) disablePlanControls(disabled bool) {
	if disabled {
		ui.btnPositive.Disable()
		ui.btnNegative.Disable()
		ui.btnAddPoint.Disable()
		ui.btnRemovePoint.Disable()
		ui.btnAddArea.Disable()
		ui.btnRemoveArea.Disable()
		ui.btnClear.Disable()
		ui.btnConfirm.Disable()
	} else {
		ui.btnPositive.Enable()
		ui.btnClear.Enable()
		ui.btnConfirm.Enable()
		ui.updateButtonStates()
	}
}

func (ui *UI) disableSchemeControls(disabled bool) {
	if disabled {
		for _, btn := range ui.schemeButtons {
			btn.Disable()
		}
		ui.btnStartFromRegion.Disable()
	} else {
		for _, btn := range ui.schemeButtons {
			btn.Enable()
		}
		ui.updateStartButtonState()
	}
}

func (ui *UI) onSpotSettingChange(spotDiameter, spotDistance, exposure int) {
	ui.spotDiameter = spotDiameter
	ui.spotDistance = spotDistance
	ui.exposureTime = exposure
	if ui.currentImage == nil || ui.originalImage == nil {
		return
	}
	if len(ui.schemes) > 0 {
		ui.rebuildSchemePlans()
		return
	}
	if ui.state.HasPlan && ui.state.CurrentMask != nil {
		ui.rebuildCurrentPlanFromMask()
	}
}

func (ui *UI) renderOverlay(overlay image.Image) {
	overlay = ui.applyFazOverlay(overlay)
	display := utils.ResizeBilinear(overlay, ui.displaySize, ui.displaySize)
	ui.imageView.Image = display
	ui.imageView.Refresh()
}

func (ui *UI) computeFazEllipse() {
	ui.fazEllipse = nil
	if ui.fazMask == nil {
		return
	}
	mask := utils.Binarize(*ui.fazMask, 1)
	boundary := extractBoundaryPoints(mask)
	if len(boundary) < 5 {
		return
	}
	ellipse := fitEllipse(boundary)
	ui.fazEllipse = &ellipse
}

func (ui *UI) applyFazOverlay(img image.Image) image.Image {
	if ui.fazEllipse == nil {
		return img
	}
	base := utils.ImageToRGBA(img)
	overlay := image.NewRGBA(base.Bounds())
	ellipse := ui.fazEllipse
	points := ellipsePoints(*ellipse, 200)
	fillPolygonRGBA(overlay, points, color.RGBA{R: 255, G: 170, B: 0, A: 77})
	drawPolygonRGBA(overlay, points, color.RGBA{R: 255, G: 170, B: 0, A: 180})
	drawCenterLabel(overlay, ellipse.Center, "禁", color.RGBA{R: 255, G: 170, B: 0, A: 200})
	combined := utils.ImageToRGBA(img)
	blendRGBA(combined, overlay)
	return combined
}

func (ui *UI) rebuildCurrentPlanFromMask() {
	if ui.originalImage == nil || ui.state.CurrentMask == nil {
		return
	}
	center := ui.ensureFAZCenter()
	plan := planner.PlanSurgery(ui.originalImage, *ui.state.CurrentMask, center, ui.areaMask, ui.spotDiameter, ui.spotDistance)
	ui.plan = &plan
	ui.renderOverlay(plan.Overlay)
}

func (ui *UI) buildSchemes(masks []utils.Mask, logits []utils.FloatMask, clicks []utils.Click) {
	if ui.originalImage == nil {
		return
	}
	center := ui.ensureFAZCenter()
	ui.schemes = nil
	for idx, mask := range masks {
		plan := planner.PlanSurgery(ui.originalImage, mask, center, ui.areaMask, ui.spotDiameter, ui.spotDistance)
		scheme := SchemeData{
			Mask:    mask,
			Plan:    plan,
			Circles: append([]image.Point(nil), plan.CircleCenters...),
			Color:   schemeColor(idx),
		}
		if idx < len(logits) {
			logitsCopy := logits[idx]
			scheme.Logits = &logitsCopy
		}
		if idx < len(clicks) {
			click := clicks[idx]
			scheme.Click = &click
		}
		ui.schemes = append(ui.schemes, scheme)
	}
	ui.refreshSchemeButtons()
	if len(ui.schemes) > 0 {
		ui.plan = &ui.schemes[0].Plan
	}
	ui.autoMode = len(ui.schemes) > 1
	if ui.autoMode {
		ui.assignRemainingCircles()
		ui.disablePlanControls(true)
	} else {
		ui.disablePlanControls(false)
	}
	ui.disableSchemeControls(false)
	ui.renderSchemeOverlay()
}

func (ui *UI) rebuildSchemePlans() {
	if ui.originalImage == nil || len(ui.schemes) == 0 {
		return
	}
	center := ui.ensureFAZCenter()
	for i := range ui.schemes {
		plan := planner.PlanSurgery(ui.originalImage, ui.schemes[i].Mask, center, ui.areaMask, ui.spotDiameter, ui.spotDistance)
		ui.schemes[i].Plan = plan
		ui.schemes[i].Circles = append([]image.Point(nil), plan.CircleCenters...)
	}
	if ui.autoMode {
		ui.assignRemainingCircles()
	}
	if len(ui.schemes) > 0 {
		ui.plan = &ui.schemes[0].Plan
	}
	ui.updateStartButtonState()
	ui.renderSchemeOverlay()
}

func (ui *UI) assignRemainingCircles() {
	if ui.originalImage == nil || ui.areaMask == nil || len(ui.schemes) == 0 {
		return
	}
	center := ui.ensureFAZCenter()
	areaPlan := planner.PlanSurgery(ui.originalImage, utils.Binarize(*ui.areaMask, 1), center, ui.areaMask, ui.spotDiameter, ui.spotDistance)
	allCenters := map[image.Point]struct{}{}
	for _, pt := range areaPlan.CircleCenters {
		allCenters[pt] = struct{}{}
	}
	assigned := map[image.Point]struct{}{}
	for _, scheme := range ui.schemes {
		for _, pt := range scheme.Circles {
			assigned[pt] = struct{}{}
		}
	}
	unassigned := make([]image.Point, 0)
	for pt := range allCenters {
		if _, ok := assigned[pt]; !ok {
			unassigned = append(unassigned, pt)
		}
	}
	if len(unassigned) == 0 {
		return
	}
	distanceMaps := make([][]float32, len(ui.schemes))
	for i, scheme := range ui.schemes {
		maskBin := utils.Binarize(scheme.Mask, 1)
		if utils.MaskArea(maskBin) == 0 {
			distanceMaps[i] = nil
			continue
		}
		inverse := utils.NotMask(maskBin)
		distanceMaps[i] = utils.DistanceTransform(inverse)
	}
	width := ui.areaMask.Width
	for _, centerPt := range unassigned {
		idx := centerPt.Y*width + centerPt.X
		bestIdx := 0
		bestDist := float32(0)
		init := false
		for i, dist := range distanceMaps {
			if dist == nil {
				continue
			}
			value := dist[idx]
			if !init || value < bestDist {
				bestDist = value
				bestIdx = i
				init = true
			}
		}
		ui.schemes[bestIdx].Circles = append(ui.schemes[bestIdx].Circles, centerPt)
	}
}

func (ui *UI) schemeLabel(index int) string {
	labels := []string{"一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五"}
	if index < len(labels) {
		return fmt.Sprintf("区域%s", labels[index])
	}
	return fmt.Sprintf("区域%d", index+1)
}

func (ui *UI) refreshSchemeButtons() {
	ui.schemeContainer.Objects = nil
	ui.schemeButtons = nil
	for idx := range ui.schemes {
		idxCopy := idx
		btn := widget.NewButton(ui.schemeLabel(idx), func() { ui.toggleSchemeSelection(idxCopy) })
		ui.schemeContainer.Add(btn)
		ui.schemeButtons = append(ui.schemeButtons, btn)
	}
	ui.schemeContainer.Refresh()
	ui.updateSchemeButtonStates()
	ui.updateStartButtonState()
}

func (ui *UI) toggleSchemeSelection(index int) {
	if ui.selectedSchemeIndex != nil && *ui.selectedSchemeIndex == index {
		ui.selectedSchemeIndex = nil
		ui.updateSchemeButtonStates()
		ui.updateStartButtonState()
		return
	}
	ui.selectedSchemeIndex = &index
	ui.updateSchemeButtonStates()
	ui.updateStartButtonState()
}

func (ui *UI) updateSchemeButtonStates() {
	for idx, btn := range ui.schemeButtons {
		if ui.selectedSchemeIndex != nil && *ui.selectedSchemeIndex == idx {
			btn.Importance = widget.HighImportance
		} else {
			btn.Importance = widget.MediumImportance
		}
		btn.Refresh()
	}
}

func (ui *UI) updateStartButtonState() {
	if !ui.autoMode || ui.selectedSchemeIndex == nil {
		ui.btnStartFromRegion.Disable()
		return
	}
	ui.btnStartFromRegion.Enable()
}

func (ui *UI) startFromRegion() {
	if ui.selectedSchemeIndex == nil || ui.originalImage == nil || len(ui.schemes) == 0 {
		return
	}
	selected := ui.schemes[*ui.selectedSchemeIndex]
	if selected.Click == nil {
		dialog.ShowInformation("提示", "未记录该区域的点击点", ui.window)
		return
	}
	ui.state.CurrentMask = &selected.Mask
	if selected.Logits != nil {
		ui.state.CurrentLogits = selected.Logits
	}
	if selected.Click != nil {
		ui.state.Clicks = []utils.Click{*selected.Click}
		ui.state.AutoClick = selected.Click
	}
	updatedMask := selected.Mask.Clone()
	circleRadius := ui.circleRadius()
	for _, center := range selected.Circles {
		if updatedMask.At(center.X, center.Y) == 0 {
			drawCircleMask(&updatedMask, center, circleRadius, 1)
		}
	}
	updatedMask = ui.applyAreaConstraint(updatedMask)
	ui.state.CurrentMask = &updatedMask
	ui.autoMode = false
	ui.schemes = nil
	ui.selectedSchemeIndex = nil
	ui.refreshSchemeButtons()
	ui.disableSchemeControls(true)
	ui.disablePlanControls(false)
	plan := planner.PlanSurgery(ui.originalImage, updatedMask, ui.ensureFAZCenter(), ui.areaMask, ui.spotDiameter, ui.spotDistance)
	ui.plan = &plan
	ui.renderOverlay(plan.Overlay)
}

func (ui *UI) renderSchemeOverlay() {
	if len(ui.schemes) == 0 {
		return
	}
	plan := ui.schemes[0].Plan
	if ui.autoMode {
		plan = ui.buildAutoOverlay(plan)
	}
	ui.plan = &plan
	ui.renderOverlay(plan.Overlay)
}

func (ui *UI) buildAutoOverlay(plan utils.PlanResult) utils.PlanResult {
	overlay := utils.ImageToRGBA(plan.Overlay)
	for idx, scheme := range ui.schemes {
		color := schemeColor(idx)
		for _, center := range scheme.Circles {
			drawCircle(overlay, center, ui.circleRadius(), color)
		}
	}
	plan.Overlay = overlay
	return plan
}

func (ui *UI) circleRadius() int {
	return int(math.Max(1, math.Round(float64(ui.spotDiameter)/2)))
}

func (ui *UI) onCanvasTap(point image.Point) {
	if ui.state.Mode == "none" {
		return
	}
	if ui.state.Mode == "add_area" || ui.state.Mode == "remove_area" {
		return
	}
	if ui.currentImage == nil || ui.originalImage == nil {
		dialog.ShowInformation("提示", "请先打开图像", ui.window)
		return
	}
	if point.X < 0 || point.Y < 0 || point.X >= ui.displaySize || point.Y >= ui.displaySize {
		return
	}
	orig := ui.displayToOriginal(point)
	click := utils.Click{X: float64(orig.X), Y: float64(orig.Y), Label: 1}
	if ui.state.Mode == "add_negative" {
		click.Label = 0
	}
	log.Printf("user_click=(%d,%d) label=%d", orig.X, orig.Y, click.Label)

	if !ui.state.HasPlan && click.Label == 1 {
		ui.state.Clicks = []utils.Click{click}
		ui.state.AutoClick = nil
		firstW, firstH := ui.pipeline.FirstModel.ImageInputSize(ui.originalImage.Bounds().Dx(), ui.originalImage.Bounds().Dy())
		firstImage := utils.PrepareImageForModel(ui.originalImage, firstW, firstH)
		result, err := ui.pipeline.FirstModel.Infer(firstImage, ui.state.Clicks, nil)
		if err != nil {
			dialog.ShowError(err, ui.window)
			return
		}
		displayMask := utils.ResizeMaskNearest(result.Mask, ui.originalImage.Bounds().Dx(), ui.originalImage.Bounds().Dy())
		displayMask = ui.postprocessFirstMask(displayMask)
		center := ui.ensureFAZCenter()
		plan := planner.PlanSurgery(ui.originalImage, displayMask, center, ui.areaMask, ui.spotDiameter, ui.spotDistance)
		ui.applyPlan(inference.Result{Mask: result.Mask, Logits: result.Logits}, plan, displayMask)
		ui.state.HasPlan = true
		ui.updateButtonStates()
		return
	}

	if !ui.state.HasPlan {
		return
	}

	switch ui.state.Mode {
	case "add_point":
		ui.applyPointModification(orig, true)
		return
	case "remove_point":
		ui.applyPointModification(orig, false)
		return
	}

	ui.state.Clicks = append(ui.state.Clicks, click)
	if ui.state.CurrentLogits == nil {
		dialog.ShowError(fmt.Errorf("缺少上一轮 logits"), ui.window)
		return
	}
	result, err := ui.pipeline.RunIteration(*ui.currentImage, ui.state.Clicks, *ui.state.CurrentLogits)
	if err != nil {
		dialog.ShowError(err, ui.window)
		return
	}
	displayMask := utils.ResizeMaskNearest(result.Mask, ui.originalImage.Bounds().Dx(), ui.originalImage.Bounds().Dy())
	displayMask = ui.applyAreaConstraint(displayMask)
	plan := planner.PlanSurgery(ui.originalImage, displayMask, ui.ensureFAZCenter(), ui.areaMask, ui.spotDiameter, ui.spotDistance)
	ui.applyPlan(result, plan, displayMask)
}

func (ui *UI) applyPlan(result inference.Result, plan utils.PlanResult, displayMask utils.Mask) {
	ui.state.CurrentMask = &displayMask
	ui.state.CurrentLogits = &result.Logits
	if len(ui.schemes) > 0 {
		ui.schemes[0].Mask = displayMask
		ui.rebuildSchemePlans()
	} else {
		ui.plan = &plan
		ui.renderOverlay(plan.Overlay)
	}
}

func (ui *UI) clearPlan() {
	if ui.originalImage == nil {
		return
	}
	ui.state = AppState{Mode: "none"}
	ui.plan = nil
	ui.schemes = nil
	ui.selectedSchemeIndex = nil
	ui.autoMode = false
	ui.refreshSchemeButtons()
	ui.disableSchemeControls(true)
	ui.updateButtonStates()
	ui.refreshToggleStates()
	ui.renderOverlay(ui.originalImage)
}

func (ui *UI) confirmPlan() {
	ui.btnPositive.Disable()
	ui.btnNegative.Disable()
	ui.btnAddPoint.Disable()
	ui.btnRemovePoint.Disable()
	ui.btnAddArea.Disable()
	ui.btnRemoveArea.Disable()
	ui.btnConfirm.Disable()
}

func (ui *UI) applyPointModification(point image.Point, add bool) {
	if ui.plan == nil || ui.currentImage == nil || ui.originalImage == nil {
		return
	}
	if ui.state.CurrentMask == nil || ui.state.CurrentLogits == nil {
		dialog.ShowError(fmt.Errorf("缺少 mask 或 logits"), ui.window)
		return
	}
	nearest, displayed := ui.findNearestCircle(point)
	if nearest == nil {
		dialog.ShowInformation("提示", "未找到可用圆圈", ui.window)
		return
	}
	if add && displayed {
		return
	}
	if !add && !displayed {
		return
	}
	value := uint8(0)
	if add {
		value = 1
	}
	ui.updateCircleMaskLogits(*nearest, value)
	updated := ui.applyAreaConstraint(*ui.state.CurrentMask)
	ui.state.CurrentMask = &updated
	plan := planner.PlanSurgery(ui.originalImage, updated, ui.ensureFAZCenter(), ui.areaMask, ui.spotDiameter, ui.spotDistance)
	if len(ui.schemes) > 0 {
		ui.schemes[0].Mask = updated
		ui.rebuildSchemePlans()
	} else {
		ui.plan = &plan
		ui.renderOverlay(plan.Overlay)
	}
}

func (ui *UI) applyAreaModification(polygon []image.Point, add bool) {
	if ui.currentImage == nil || ui.originalImage == nil {
		return
	}
	if ui.state.CurrentMask == nil || ui.state.CurrentLogits == nil {
		dialog.ShowError(fmt.Errorf("缺少 mask 或 logits"), ui.window)
		return
	}
	maskUpdate := utils.NewMask(ui.state.CurrentMask.Width, ui.state.CurrentMask.Height)
	fillPolygonMask(&maskUpdate, polygon, 1)
	updated := ui.state.CurrentMask.Clone()
	if add {
		updated = utils.OrMask(updated, maskUpdate)
	} else {
		updated = utils.AndMask(updated, utils.NotMask(maskUpdate))
	}
	updated = ui.applyAreaConstraint(updated)
	ui.state.CurrentMask = &updated
	ui.updateLogitsWithPolygon(polygon, add)
	plan := planner.PlanSurgery(ui.originalImage, updated, ui.ensureFAZCenter(), ui.areaMask, ui.spotDiameter, ui.spotDistance)
	if len(ui.schemes) > 0 {
		ui.schemes[0].Mask = updated
		ui.rebuildSchemePlans()
	} else {
		ui.plan = &plan
		ui.renderOverlay(plan.Overlay)
	}
}

func (ui *UI) postprocessFirstMask(mask utils.Mask) utils.Mask {
	cleaned := utils.RemoveSmallComponents(mask, 400)
	filled := utils.FillSmallHoles(cleaned, 400)
	if ui.areaMask != nil {
		areaBin := utils.Binarize(*ui.areaMask, 1)
		filled = utils.AndMask(filled, areaBin)
	}
	return filled
}

func (ui *UI) applyAreaConstraint(mask utils.Mask) utils.Mask {
	if ui.areaMask == nil {
		return mask
	}
	areaBin := utils.Binarize(*ui.areaMask, 1)
	return utils.AndMask(mask, areaBin)
}

func (ui *UI) findNearestCircle(point image.Point) (*image.Point, bool) {
	if ui.plan == nil || ui.originalImage == nil {
		return nil, false
	}
	allPoints := make([]image.Point, 0)
	for _, ring := range ui.plan.CurvePoints {
		for _, pt := range ring {
			if pt.X >= 0 && pt.Y >= 0 && pt.X < ui.originalImage.Bounds().Dx() && pt.Y < ui.originalImage.Bounds().Dy() {
				allPoints = append(allPoints, pt)
			}
		}
	}
	if len(allPoints) == 0 {
		return nil, false
	}
	minIdx := 0
	minDist := math.MaxFloat64
	for idx, pt := range allPoints {
		dx := float64(pt.X - point.X)
		dy := float64(pt.Y - point.Y)
		d := dx*dx + dy*dy
		if d < minDist {
			minDist = d
			minIdx = idx
		}
	}
	nearest := allPoints[minIdx]
	displayed := false
	for _, pt := range ui.plan.CircleCenters {
		if pt == nearest {
			displayed = true
			break
		}
	}
	return &nearest, displayed
}

func (ui *UI) updateCircleMaskLogits(center image.Point, value uint8) {
	if ui.state.CurrentMask == nil || ui.state.CurrentLogits == nil {
		return
	}
	maskCircle := utils.NewMask(ui.state.CurrentMask.Width, ui.state.CurrentMask.Height)
	drawCircleMask(&maskCircle, center, ui.circleRadius(), 1)
	updated := ui.state.CurrentMask.Clone()
	if value == 1 {
		updated = utils.OrMask(updated, maskCircle)
	} else {
		updated = utils.AndMask(updated, utils.NotMask(maskCircle))
	}
	ui.state.CurrentMask = &updated

	logits := ui.state.CurrentLogits.Clone()
	scaleX := float64(logits.Width) / float64(ui.originalImage.Bounds().Dx())
	scaleY := float64(logits.Height) / float64(ui.originalImage.Bounds().Dy())
	logitsCenter := image.Point{X: int(float64(center.X) * scaleX), Y: int(float64(center.Y) * scaleY)}
	logitsRadius := int(math.Max(1, math.Round(float64(ui.circleRadius())*(scaleX+scaleY)/2)))
	logitsCircle := utils.NewMask(logits.Width, logits.Height)
	drawCircleMask(&logitsCircle, logitsCenter, logitsRadius, 1)
	for i, v := range logitsCircle.Data {
		if v == 0 {
			continue
		}
		if value == 1 {
			logits.Data[i] = 1
		} else {
			logits.Data[i] = 0
		}
	}
	ui.state.CurrentLogits = &logits
}

func (ui *UI) updateLogitsWithPolygon(polygon []image.Point, add bool) {
	if ui.state.CurrentLogits == nil || ui.originalImage == nil {
		return
	}
	logits := ui.state.CurrentLogits.Clone()
	scaleX := float64(logits.Width) / float64(ui.originalImage.Bounds().Dx())
	scaleY := float64(logits.Height) / float64(ui.originalImage.Bounds().Dy())
	scaled := make([]image.Point, len(polygon))
	for i, pt := range polygon {
		scaled[i] = image.Point{X: int(float64(pt.X) * scaleX), Y: int(float64(pt.Y) * scaleY)}
	}
	maskLogits := utils.NewMask(logits.Width, logits.Height)
	fillPolygonMask(&maskLogits, scaled, 1)
	for i, v := range maskLogits.Data {
		if v == 0 {
			continue
		}
		if add {
			logits.Data[i] = 1
		} else {
			logits.Data[i] = 0
		}
	}
	ui.state.CurrentLogits = &logits
}

func (ui *UI) ensureFAZCenter() image.Point {
	if ui.fazCenter != nil {
		return *ui.fazCenter
	}
	center := image.Point{X: ui.originalImage.Bounds().Dx() / 2, Y: ui.originalImage.Bounds().Dy() / 2}
	ui.fazCenter = &center
	return center
}

func (ui *UI) displayToOriginal(point image.Point) image.Point {
	return image.Point{X: int(float64(point.X) * ui.displayScaleX), Y: int(float64(point.Y) * ui.displayScaleY)}
}

func (ui *UI) startPolygon(point image.Point) {
	ui.drawingPoints = []image.Point{point}
}

func (ui *UI) extendPolygon(point image.Point) {
	ui.drawingPoints = append(ui.drawingPoints, point)
}

func (ui *UI) finishPolygon() {
	if len(ui.drawingPoints) < 3 {
		ui.drawingPoints = nil
		return
	}
	polygon := make([]image.Point, len(ui.drawingPoints))
	for i, p := range ui.drawingPoints {
		polygon[i] = ui.displayToOriginal(p)
	}
	if ui.state.Mode == "add_area" {
		ui.applyAreaModification(polygon, true)
	} else if ui.state.Mode == "remove_area" {
		ui.applyAreaModification(polygon, false)
	}
	ui.drawingPoints = nil
}

// overlay widget

type imageOverlay struct {
	widget.BaseWidget
	ui *UI
}

func newImageOverlay(ui *UI) *imageOverlay {
	o := &imageOverlay{ui: ui}
	o.ExtendBaseWidget(o)
	return o
}

func (o *imageOverlay) CreateRenderer() fyne.WidgetRenderer {
	rect := canvas.NewRectangle(color.Transparent)
	return widget.NewSimpleRenderer(rect)
}

func (o *imageOverlay) Tapped(ev *fyne.PointEvent) {
	point := image.Point{X: int(ev.Position.X), Y: int(ev.Position.Y)}
	o.ui.onCanvasTap(point)
}

func (o *imageOverlay) Dragged(ev *fyne.DragEvent) {
	if o.ui.state.Mode != "add_area" && o.ui.state.Mode != "remove_area" {
		return
	}
	point := image.Point{X: int(ev.Position.X), Y: int(ev.Position.Y)}
	if len(o.ui.drawingPoints) == 0 {
		o.ui.startPolygon(point)
		return
	}
	o.ui.extendPolygon(point)
}

func (o *imageOverlay) DragEnd() {
	if o.ui.state.Mode != "add_area" && o.ui.state.Mode != "remove_area" {
		return
	}
	o.ui.finishPolygon()
}

func (o *imageOverlay) MouseIn(*desktop.MouseEvent) {
	o.ui.mouseOver = true
}

func (o *imageOverlay) MouseOut() {
	o.ui.mouseOver = false
	o.ui.lastMousePos = nil
}

func (o *imageOverlay) MouseMoved(ev *desktop.MouseEvent) {
	if o.ui.state.Mode == "add_positive" || o.ui.state.Mode == "add_negative" {
		point := image.Point{X: int(ev.Position.X), Y: int(ev.Position.Y)}
		if point.X >= 0 && point.Y >= 0 && point.X < o.ui.displaySize && point.Y < o.ui.displaySize {
			o.ui.lastMousePos = &point
		}
	}
}

// geometry and drawing helpers

func schemeColor(index int) color.RGBA {
	colors := []color.RGBA{
		{R: 0, G: 112, B: 255, A: 255},
		{R: 220, G: 20, B: 60, A: 255},
		{R: 60, G: 179, B: 113, A: 255},
		{R: 138, G: 43, B: 226, A: 255},
		{R: 0, G: 206, B: 209, A: 255},
		{R: 199, G: 21, B: 133, A: 255},
		{R: 160, G: 82, B: 45, A: 255},
		{R: 128, G: 0, B: 0, A: 255},
		{R: 0, G: 128, B: 128, A: 255},
		{R: 0, G: 0, B: 0, A: 255},
	}
	return colors[index%len(colors)]
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

func drawCircleMask(mask *utils.Mask, center image.Point, radius int, value uint8) {
	for y := center.Y - radius; y <= center.Y+radius; y++ {
		for x := center.X - radius; x <= center.X+radius; x++ {
			dx := x - center.X
			dy := y - center.Y
			if dx*dx+dy*dy > radius*radius {
				continue
			}
			mask.Set(x, y, value)
		}
	}
}

func fillPolygonMask(mask *utils.Mask, polygon []image.Point, value uint8) {
	if len(polygon) < 3 {
		return
	}
	minY, maxY := polygon[0].Y, polygon[0].Y
	for _, pt := range polygon {
		if pt.Y < minY {
			minY = pt.Y
		}
		if pt.Y > maxY {
			maxY = pt.Y
		}
	}
	for y := minY; y <= maxY; y++ {
		intersections := make([]int, 0)
		for i := 0; i < len(polygon); i++ {
			j := (i + 1) % len(polygon)
			p1 := polygon[i]
			p2 := polygon[j]
			if p1.Y == p2.Y {
				continue
			}
			if (y < p1.Y && y < p2.Y) || (y > p1.Y && y > p2.Y) {
				continue
			}
			x := p1.X + int(float64(y-p1.Y)*float64(p2.X-p1.X)/float64(p2.Y-p1.Y))
			intersections = append(intersections, x)
		}
		if len(intersections) < 2 {
			continue
		}
		sort.Ints(intersections)
		for i := 0; i < len(intersections); i += 2 {
			if i+1 >= len(intersections) {
				break
			}
			x1 := intersections[i]
			x2 := intersections[i+1]
			if x1 > x2 {
				x1, x2 = x2, x1
			}
			for x := x1; x <= x2; x++ {
				mask.Set(x, y, value)
			}
		}
	}
}

func extractBoundaryPoints(mask utils.Mask) []image.Point {
	points := make([]image.Point, 0)
	for y := 1; y < mask.Height-1; y++ {
		for x := 1; x < mask.Width-1; x++ {
			if mask.At(x, y) == 0 {
				continue
			}
			if mask.At(x-1, y) == 0 || mask.At(x+1, y) == 0 || mask.At(x, y-1) == 0 || mask.At(x, y+1) == 0 {
				points = append(points, image.Point{X: x, Y: y})
			}
		}
	}
	return points
}

func fitEllipse(points []image.Point) ellipseInfo {
	var sumX, sumY float64
	for _, pt := range points {
		sumX += float64(pt.X)
		sumY += float64(pt.Y)
	}
	cx := sumX / float64(len(points))
	cy := sumY / float64(len(points))
	var sxx, sxy, syy float64
	for _, pt := range points {
		dx := float64(pt.X) - cx
		dy := float64(pt.Y) - cy
		sxx += dx * dx
		sxy += dx * dy
		syy += dy * dy
	}
	sxx /= float64(len(points))
	sxy /= float64(len(points))
	syy /= float64(len(points))
	trace := sxx + syy
	det := sxx*syy - sxy*sxy
	lambda1 := trace/2 + math.Sqrt(math.Max(0, trace*trace/4-det))
	lambda2 := trace/2 - math.Sqrt(math.Max(0, trace*trace/4-det))
	angle := 0.5 * math.Atan2(2*sxy, sxx-syy)
	axis1 := math.Sqrt(math.Max(lambda1, 0))
	axis2 := math.Sqrt(math.Max(lambda2, 0))
	cosA := math.Cos(angle)
	sinA := math.Sin(angle)
	maxX, maxY := 0.0, 0.0
	for _, pt := range points {
		dx := float64(pt.X) - cx
		dy := float64(pt.Y) - cy
		xr := dx*cosA + dy*sinA
		yr := -dx*sinA + dy*cosA
		if math.Abs(xr) > maxX {
			maxX = math.Abs(xr)
		}
		if math.Abs(yr) > maxY {
			maxY = math.Abs(yr)
		}
	}
	width := math.Max(2*maxX, 2*axis1)
	height := math.Max(2*maxY, 2*axis2)
	return ellipseInfo{Center: image.Point{X: int(cx), Y: int(cy)}, Width: width, Height: height, Angle: angle}
}

func ellipsePoints(e ellipseInfo, steps int) []image.Point {
	points := make([]image.Point, 0, steps)
	rx := e.Width / 2
	ry := e.Height / 2
	cosA := math.Cos(e.Angle)
	sinA := math.Sin(e.Angle)
	for i := 0; i < steps; i++ {
		t := 2 * math.Pi * float64(i) / float64(steps)
		x := rx * math.Cos(t)
		y := ry * math.Sin(t)
		xr := x*cosA - y*sinA
		yr := x*sinA + y*cosA
		points = append(points, image.Point{X: e.Center.X + int(math.Round(xr)), Y: e.Center.Y + int(math.Round(yr))})
	}
	return points
}

func fillPolygonRGBA(img *image.RGBA, points []image.Point, fill color.RGBA) {
	mask := utils.NewMask(img.Bounds().Dx(), img.Bounds().Dy())
	fillPolygonMask(&mask, points, 1)
	for i, v := range mask.Data {
		if v == 0 {
			continue
		}
		x := i % mask.Width
		y := i / mask.Width
		img.SetRGBA(x, y, fill)
	}
}

func drawPolygonRGBA(img *image.RGBA, points []image.Point, stroke color.RGBA) {
	for i := 0; i < len(points); i++ {
		j := (i + 1) % len(points)
		drawLine(img, points[i], points[j], stroke)
	}
}

func drawLine(img *image.RGBA, p1, p2 image.Point, stroke color.RGBA) {
	dx := math.Abs(float64(p2.X - p1.X))
	dy := math.Abs(float64(p2.Y - p1.Y))
	sx := -1
	if p1.X < p2.X {
		sx = 1
	}
	sy := -1
	if p1.Y < p2.Y {
		sy = 1
	}
	err := dx - dy
	x, y := p1.X, p1.Y
	for {
		if x >= 0 && y >= 0 && x < img.Bounds().Dx() && y < img.Bounds().Dy() {
			img.SetRGBA(x, y, stroke)
		}
		if x == p2.X && y == p2.Y {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}
}

func blendRGBA(base *image.RGBA, overlay *image.RGBA) {
	for y := 0; y < base.Bounds().Dy(); y++ {
		for x := 0; x < base.Bounds().Dx(); x++ {
			br, bg, bb, ba := base.At(x, y).RGBA()
			or, og, ob, oa := overlay.At(x, y).RGBA()
			alpha := float64(oa) / 65535.0
			inv := 1 - alpha
			r := uint8((float64(br>>8)*inv + float64(or>>8)*alpha))
			g := uint8((float64(bg>>8)*inv + float64(og>>8)*alpha))
			b := uint8((float64(bb>>8)*inv + float64(ob>>8)*alpha))
			base.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: uint8(ba >> 8)})
		}
	}
}

func drawCenterLabel(img *image.RGBA, center image.Point, text string, tint color.RGBA) {
	if text == "" {
		return
	}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(tint),
		Face: basicfont.Face7x13,
	}
	bounds := d.MeasureString(text)
	width := bounds.Round()
	d.Dot = fixed.Point26_6{
		X: fixed.I(center.X - width/2),
		Y: fixed.I(center.Y),
	}
	d.DrawString(text)
}
