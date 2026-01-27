module eyesam/standalone_app_v3_5_go

go 1.22

require (
	fyne.io/fyne/v2 v2.5.3
	github.com/yalue/onnxruntime_go v0.0.0-00010101000000-000000000000
	golang.org/x/image v0.20.0
)

replace github.com/yalue/onnxruntime_go => ../onnxruntime_go-master
