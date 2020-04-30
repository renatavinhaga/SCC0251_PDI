import numpy as np 
#import matplotlib.pyplot as pt 
import imageio

def compare(m, r):
	return np.sqrt(np.sum((m - r)**2))

def method2(f, kernel, c):
	N,M = f.shape #dimension of f
	n,m = 3,3
	a = int((n-1)/2)
	b = int((n-1)/2)
	f = np.pad(f, [(1, 1), (1, 1)], mode='constant', constant_values=0)
	I = np.zeros(f.shape)
    
	if kernel == 1:
		k = np.matrix([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
	else:	
		k = np.matrix([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
	
	for x in range(a, N+1):
		for y in range(b, M+1):
			region_f = f[ x-a : x + (a+1), y-b : y+(b+1) ]
			I[x,y] = np.sum(np.multiply(k, region_f))

	f = f[1:f.shape[0]-1, 1:f.shape[1]-1]	
	I = I[1:I.shape[0]-1, 1:I.shape[1]-1]	

	min_I = I.min()
	max_I = I.max()
	I = ((I - min_I) * 255)/(max_I)
	r = c * I + f
	return ((r - r.min()) * 255)/(r.max())

filename = str(input()).rstrip()
m = imageio.imread("imgs/" + filename)
method = int(input())
save = int(input())

m = np.asarray(m, dtype = float)

if method == 1:
	n = int(input())
	s = float(input())
	r = float(input())
if method == 2:
	# m = np.matrix([[23,    5,  39,  45,  50],
	#                [70,   88,  12, 100, 110],
	#                [130, 145, 159, 136, 137],
	#                [19,  200, 201, 220, 203],
	#                [25,   26,  27,  28, 209],
	#                [131,  32, 133,  34, 135]])
	c = float(input())
	kernel = int(input())
	r = method2(m, kernel, c)

if method == 3:
	kernel = int(input())
	row = int(input())
	col = int(input())

output_img = r
print("{:.4f}".format(compare(m,r)))

output_img = np.asarray(output_img, dtype="uint8")

if save == 1:
	imageio.imwrite('output_img.png', output_img)