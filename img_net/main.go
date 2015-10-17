package main

import (
	"fmt"
	"image"
	"image/color"
	//"math"
	"math/rand"

	"github.com/avesanen/neuro"
	"github.com/disintegration/imaging"
)

type ImgNet struct {
	w, h int
	net  *neuro.Network
}

func NewImgNet(w, h int) *ImgNet {
	topology := []int{w * h * 3, 9, w * h * 3}
	i := &ImgNet{
		w:   w,
		h:   h,
		net: neuro.NewNetwork(topology),
	}
	return i
}

func (i *ImgNet) cut(p image.Point, img *image.Image) []float64 {
	rect := image.Rect(p.X, p.Y, p.X+i.w, p.Y+i.h)
	croppedImg := imaging.Crop(*img, rect)

	rgb := make([]float64, i.w*i.h*3)

	for y := 0; y < croppedImg.Bounds().Dy(); y++ {
		for x := 0; x < croppedImg.Bounds().Dx(); x++ {
			r, g, b, _ := croppedImg.At(x, y).RGBA()
			index := x + y*i.w*3
			rgb[index] = float64(r) / 65535
			rgb[index+1] = float64(g) / 65535
			rgb[index+2] = float64(b) / 65535
		}
	}
	return rgb
}

func (i *ImgNet) FF(p image.Point, img *image.Image) {
	rgb := i.cut(p, img)
	i.net.FeedForward(rgb)
}

func (i *ImgNet) BP(p image.Point, img *image.Image) {
	rgb := i.cut(p, img)
	i.net.BackPropagation(rgb)
}

func (i *ImgNet) Filter(img image.Image) image.Image {
	filtered := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
	// Move network vision
	for y := -i.h; y < img.Bounds().Dy(); y++ {
		for x := -i.w; x < img.Bounds().Dy(); x++ {
			rgb := i.cut(image.Pt(x, y), &img)
			i.net.FeedForward(rgb)

			result := i.net.Results()
			r := uint8(result[i.w/2+(i.h/2*i.w)] * 255)
			g := uint8(result[i.w/2+(i.h/2*i.w)+(i.w*i.h)] * 255)
			b := uint8(result[i.w/2+(i.h/2*i.w)+(i.w*i.h*2)] * 255)
			c := color.RGBA{r, g, b, 255}
			filtered.Set(x, y, c)
		}
	}
	return filtered
}

func main() {
	fmt.Println("ImgNet loading...")
	netSize := []int{3, 3}
	net := NewImgNet(netSize[0], netSize[1])

	imgOrig, err := imaging.Open("lena-colour.png")
	if err != nil {
		panic(err)
	}

	imgFiltered, err := imaging.Open("lena-colourdiff.png")
	if err != nil {
		panic(err)
	}

	targetImg, err := imaging.Open("lena-colour.png")
	if err != nil {
		panic(err)
	}

	fmt.Println("Teaching network.")
	for randcount := 0; randcount < 1000000; randcount++ {
		x := rand.Intn(imgOrig.Bounds().Dx()-netSize[0]) - netSize[0]/2
		y := rand.Intn(imgOrig.Bounds().Dy()-netSize[1]) - netSize[1]/2
		net.FF(image.Pt(x, y), &imgOrig)
		net.BP(image.Pt(x, y), &imgFiltered)
	}

	fmt.Println("Filtering image.")
	result := net.Filter(targetImg)
	imaging.Save(result, "output.png")

}
