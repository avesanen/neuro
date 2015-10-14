package main

import (
	"image"
	"image/color"

	"github.com/avesanen/neuro"
	"github.com/disintegration/imaging"
)

func main() {

	// input files
	files := []string{"01.png", "02.png", "03.png"}

	// load images and make 100x100 thumbnails of them
	var images []image.Image
	for _, file := range files {
		img, err := imaging.Open(file)
		if err != nil {
			panic(err)
		}
		images = append(images, img)
	}

	n := neuro.NewNetwork([]int{3, 3, 3})

	for y := 0; y < images[0].Bounds().Dy(); y++ {
		for x := 0; x < images[0].Bounds().Dx(); x++ {
			imgR, imgG, imgB, _ := images[0].At(x, y).RGBA()
			r := float64(imgR) / 65535
			g := float64(imgG) / 65535
			b := float64(imgB) / 65535
			n.FeedForward([]float64{r, g, b})

			imgR2, imgG2, imgB2, _ := images[1].At(x, y).RGBA()
			r2 := float64(imgR2) / 65535
			g2 := float64(imgG2) / 65535
			b2 := float64(imgB2) / 65535
			n.BackPropagation([]float64{r2, g2, b2})
		}
	}

	m := image.NewRGBA(image.Rect(0, 0, 128, 128))
	for y := 0; y < images[2].Bounds().Dy(); y++ {
		for x := 0; x < images[2].Bounds().Dx(); x++ {
			imgR, imgG, imgB, _ := images[2].At(x, y).RGBA()
			r := float64(imgR) / 65535
			g := float64(imgG) / 65535
			b := float64(imgB) / 65535
			n.FeedForward([]float64{r, g, b})
			results := n.Results()
			c := color.RGBA{uint8(results[0] * 255), uint8(results[1] * 255), uint8(results[2] * 255), 255}
			m.Set(x, y, c)
		}
	}

	dst := imaging.New(256, 256, color.NRGBA{0, 0, 0, 0})
	dst = imaging.Paste(dst, images[0], image.Pt(0, 0))
	dst = imaging.Paste(dst, images[1], image.Pt(128, 0))
	dst = imaging.Paste(dst, images[2], image.Pt(0, 128))
	dst = imaging.Paste(dst, m, image.Pt(128, 128))

	// save the combined image to file
	err := imaging.Save(dst, "dst.jpg")
	if err != nil {
		panic(err)
	}
}
