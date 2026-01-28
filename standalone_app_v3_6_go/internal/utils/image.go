package utils

import (
	"fmt"
	"image"
	"image/color"
	imagedraw "image/draw"
	"image/jpeg"
	"image/png"
	"os"

	xdraw "golang.org/x/image/draw"
)

func LoadImage(path string) (img image.Image, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer func() {
		if cerr := file.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close image: %w", cerr)
		}
	}()

	img, _, err = image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

func SavePNG(path string, img image.Image) (err error) {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create image: %w", err)
	}
	defer func() {
		if cerr := file.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close image: %w", cerr)
		}
	}()

	if err := png.Encode(file, img); err != nil {
		return fmt.Errorf("encode png: %w", err)
	}
	return nil
}

func SaveJPEG(path string, img image.Image, quality int) (err error) {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create image: %w", err)
	}
	defer func() {
		if cerr := file.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("close image: %w", cerr)
		}
	}()

	if err := jpeg.Encode(file, img, &jpeg.Options{Quality: quality}); err != nil {
		return fmt.Errorf("encode jpeg: %w", err)
	}
	return nil
}

func ResizeNearest(src image.Image, targetWidth, targetHeight int) image.Image {
	if targetWidth <= 0 || targetHeight <= 0 {
		return src
	}
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	if w == 0 || h == 0 {
		return src
	}
	out := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	for y := 0; y < targetHeight; y++ {
		sy := int(float64(y) * float64(h) / float64(targetHeight))
		for x := 0; x < targetWidth; x++ {
			sx := int(float64(x) * float64(w) / float64(targetWidth))
			out.Set(x, y, src.At(bounds.Min.X+sx, bounds.Min.Y+sy))
		}
	}
	return out
}

func ResizeBilinear(src image.Image, targetWidth, targetHeight int) image.Image {
	if targetWidth <= 0 || targetHeight <= 0 {
		return src
	}
	bounds := src.Bounds()
	if bounds.Dx() == 0 || bounds.Dy() == 0 {
		return src
	}
	out := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	xdraw.ApproxBiLinear.Scale(out, out.Bounds(), src, bounds, xdraw.Over, nil)
	return out
}

func PrepareImageForModel(img image.Image, targetWidth, targetHeight int) ModelImage {
	resized := ResizeBilinear(img, targetWidth, targetHeight)
	scaleX := float64(targetWidth) / float64(img.Bounds().Dx())
	scaleY := float64(targetHeight) / float64(img.Bounds().Dy())
	return ModelImage{
		Original: img,
		Resized:  resized,
		ScaleX:   scaleX,
		ScaleY:   scaleY,
	}
}

func ImageToRGBA(img image.Image) *image.RGBA {
	bounds := img.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	imagedraw.Draw(rgba, rgba.Bounds(), img, bounds.Min, imagedraw.Src)
	return rgba
}

func ImageToRGBBytes(img image.Image) []uint8 {
	rgba := ImageToRGBA(img)
	data := make([]uint8, rgba.Bounds().Dx()*rgba.Bounds().Dy()*3)
	idx := 0
	for y := 0; y < rgba.Bounds().Dy(); y++ {
		for x := 0; x < rgba.Bounds().Dx(); x++ {
			r, g, b, _ := rgba.At(x, y).RGBA()
			data[idx] = uint8(r >> 8)
			data[idx+1] = uint8(g >> 8)
			data[idx+2] = uint8(b >> 8)
			idx += 3
		}
	}
	return data
}

func GrayMaskToImage(mask Mask) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, mask.Width, mask.Height))
	copy(img.Pix, mask.Data)
	return img
}

func ApplyMaskColor(img image.Image, mask Mask, tint color.RGBA) image.Image {
	base := ImageToRGBA(img)
	bounds := base.Bounds()
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			if mask.At(x, y) == 0 {
				continue
			}
			r, g, b, a := base.At(x, y).RGBA()
			base.SetRGBA(x, y, color.RGBA{
				R: uint8((uint32(r>>8) + uint32(tint.R)) / 2),
				G: uint8((uint32(g>>8) + uint32(tint.G)) / 2),
				B: uint8((uint32(b>>8) + uint32(tint.B)) / 2),
				A: uint8(a >> 8),
			})
		}
	}
	return base
}
