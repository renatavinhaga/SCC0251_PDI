#Renata Vinhaga dos Anjos 10295263
import numpy as np 
import imageio


def E(x, y):
	return np.sqrt(x*x + y*y)

def G(x, Sigma):
	return (np.exp(-(x * x)/(2 * Sigma * Sigma))/(2 * np.pi * Sigma * Sigma))

def padding(m, add):
	return np.pad(m, [(add, add), (add, add)], mode='constant', constant_values=0)

def unpadding(m, sub):
	return m[sub:m.shape[0]-sub, sub:m.shape[1]-sub]

def compare(m, r):
	return np.sqrt(np.sum((m - r)**2))

def Spatial_Gaussian(SigmaS, n):
	a = -int((n-1)/2)
	b = int((n-1)/2)
	Gs = np.zeros((n,n))
	for x in range(a,b + 1):
		for y in range(a,b + 1):
			Gs[x-a][y-a] = G(E(x,y), SigmaS)

	return Gs

def Range_Gaussian(m, SigmaR, n):
	a = -int((n-1)/2)
	b = int((n-1)/2)
	center = m[b][b]
	Gs = np.zeros((n,n))
	for x in range(0,n):
		for y in range(0, n):
			Gs[x][y] = G(m[x][y] - center, SigmaR) 
	return Gs

def method1(I, n, SigmaS, SigmaR):
	N,M = I.shape #dimension of f
	Wp = np.zeros((n,n))
	a = int((n-1)/2)
	b = int((n-1)/2)
	I = padding(I, a)
	If = np.zeros(I.shape)
	Gs = Spatial_Gaussian(SigmaS, n)
	print(Gs)
	W = 0
	for x in range(a, N+1):
		for y in range(b, M+1):
			region_f = I[x-a : x + (a+1), y-b : y+(b+1)]
			Gr = Range_Gaussian(region_f, SigmaR, n)
			Wp = Gr * Gs
			W = np.sum(Wp)
			If[x,y] = np.sum(np.multiply(Wp, region_f))
			If[x,y] = If[x,y]/W
			

	I = unpadding(I, a)
	If = unpadding(If, a)

	return If


def method2(f, kernel, c):
	N,M = f.shape #dimension of f
	n,m = 3,3
	a = int((n-1)/2)
	b = int((n-1)/2)
	f = padding(f)
	I = np.zeros(f.shape)
    
	if kernel == 1:
		k = np.matrix([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
	else:	
		k = np.matrix([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
	
	for x in range(a, N+1):
		for y in range(b, M+1):
			region_f = f[x-a : x + (a+1), y-b : y+(b+1)]
			I[x,y] = np.sum(np.multiply(k, region_f))

	f = unpadding(f)
	I = unpadding(I)	

	min_I = I.min()
	max_I = I.max()
	I = ((I - min_I) * 255)/(max_I)
	r = c * I + f
	return ((r - r.min()) * 255)/(r.max() - r.min())

def main():
	filename = str(input()).rstrip()
	m = imageio.imread("imgs/" + filename)
	method = int(input())
	save = int(input())
	
	m = np.asarray(m, dtype = float)

	if method == 1:
		n = int(input())
		s = float(input())
		r = float(input())
		output_img = method1(m, n, s, r)

	if method == 2:
		c = float(input())
		kernel = int(input())
		output_img = method2(m, kernel, c)

	if method == 3:
		kernel = int(input())
		row = int(input())
		col = int(input())
		output_img = method3()

	print(m.shape)
	print(output_img.shape)
	print("{:.4f}".format(compare(m,output_img)))

	output_img = np.asarray(output_img, dtype="uint8")

	if save == 1:
		imageio.imwrite('output_img.png', output_img)


if __name__ == "__main__":
	main()