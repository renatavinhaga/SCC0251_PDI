import imageio
import numpy as np

def inversion(img):
	return 255 - img

def contrast(img, c, d):
	min_img = img.min()
	max_img = img.max()
	return (img - min_img) * ((d - c)/(max_img - min_img)) + c

def logarithmic(img):
	return 255 * (np.log2(1 + img)/np.log2(1 + img.max()))

def gamma(img, W, lambd):
	return W * img**lambd

def compare(m, r):
	return np.sqrt(np.sum((m - r)**2))

filename = str(input()).rstrip()
m = imageio.imread(filename)
method = int(input())
save = int(input())

m = np.asarray(m, dtype = float)

if method == 1:
	r = inversion(m)
if method == 2:
	c = int(input())
	d = int(input())
	r = contrast(m, c, d)

if method == 3:
	r = logarithmic(m)

if method == 4:
	W = int(input())
	lambd = float(input())
	r = gamma(m, W, lambd)

print("{:.4f}".format(compare(m,r)))

if save == 1:
imageio.imwrite('output_img.png',output_img)