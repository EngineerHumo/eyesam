package utils

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
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
