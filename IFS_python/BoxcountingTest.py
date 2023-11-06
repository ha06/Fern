import numpy as np
import pylab as pl


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


image = rgb2gray(pl.imread("fern.png"))

# finding all the non-zero pixels
pixels = []
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] > 0:
            pixels.append((i, j))

Lx = image.shape[1]
Ly = image.shape[0]
print(Lx, Ly)
pixels = pl.array(pixels)
print(pixels.shape)

# computing the fractal dimension
# considering only scales in a logarithmic list
scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
Ns = []
# looping over several scales
for scale in scales:
    print("======= Scale :", scale)
    # computing the histogram
    H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
    Ns.append(np.sum(H > 0))

# linear fit, polynomial of degree 1
coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

pl.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
pl.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
pl.xlabel('log $\epsilon$')
pl.ylabel('log N')
pl.savefig('Hausdorff_dimension.pdf')

print("The Hausdorff dimension is", -coeffs[0])  # the fractal dimension is the OPPOSITE of the fitting coefficient
np.savetxt("scaling.txt", list(zip(scales, Ns)))