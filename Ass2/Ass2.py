#SCC0251 - Prof. Moacir Ponti
#Assignment 2 : Image Enhancement and Filtering
#Renata Vinhaga dos Anjos 10295263
#github repository link: https://github.com/renatavinhaga/SCC0251_PDI/tree/master/Ass2

import numpy as np 
import imageio

#Euclidian Distance between x and y
def E(x, y):
	return np.sqrt(x*x + y*y)

# 1D Gaussian
def G(x, Sigma):
	return (np.exp(-(x * x)/(2 * Sigma * Sigma))/(2 * np.pi * Sigma * Sigma))

#add 0 columms and rows to the matrix m according to the parameter 'add'
def padding(m, add):
	return np.pad(m, [(add, add), (add, add)], mode='constant', constant_values=0)

#remove the added 0 columns and rows
def unpadding(m, sub):
	return m[sub:m.shape[0]-sub, sub:m.shape[1]-sub]

#Calculate de difference between the original picture and new one
def Error(m, r):
	return np.sqrt(np.sum((m - r)**2))

#normalize the matrix m (0 - 255)
def normalize(m):
	return ((m - m.min()) * 255)/(m.max() - m.min())

# Return the Spatial Gaussian Kernel 
def Spatial_Gaussian(SigmaS, n):
	a = -int((n-1)/2)
	b = int((n-1)/2)
	Gs = np.zeros((n,n))
	for x in range(a,b + 1):
		for y in range(a,b + 1):
			Gs[x-a][y-a] = G(E(x,y), SigmaS)

	return Gs

# Return the Range Gaussian Kernel 
def Range_Gaussian(m, SigmaR, n):
	a = -int((n-1)/2)
	b = int((n-1)/2)
	center = m[b][b]
	Gs = np.zeros((n,n))
	for x in range(0,n):
		for y in range(0, n):
			Gs[x][y] = G(m[x][y] - center, SigmaR) 
	return Gs

#split a region of the matrix m to apply the filter desired
def cut_region(m,x,y,a,b):
	return m[x-a : x + (a+1), y-b : y+(b+1)];

# Apply the Bilateral Filter
def method1(I, n, SigmaS, SigmaR):
	N,M = I.shape #dimension of f
	Wp = np.zeros((n,n))
	a = int((n-1)/2)
	b = int((n-1)/2)
	I = padding(I, a)
	If = np.zeros(I.shape)
	Gs = Spatial_Gaussian(SigmaS, n)
	W = 0
	
	for x in range(a, N+1):
		for y in range(b, M+1):
			region = cut_region(I,x,y,a,b)
			Gr = Range_Gaussian(region, SigmaR, n)
			Wp = Gr * Gs
			W = np.sum(Wp)
			If[x,y] = np.sum(np.multiply(Wp, region))
			If[x,y] = If[x,y]/W
			

	I = unpadding(I, a)
	If = unpadding(If, a)

	return If

# Unsharp mask using the Laplacian Filter
def method2(f, kernel, c):
	N,M = f.shape #dimension of f
	n,m = 3,3
	a = int((n-1)/2)
	b = int((n-1)/2)
	f = padding(f, a)
	I = np.zeros(f.shape)
	#choose the kernel
	if kernel == 1:
		k = np.matrix([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
	else:	
		k = np.matrix([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
	
	for x in range(a, N+1):
		for y in range(b, M+1):
			region = cut_region(f,x,y,a,b)
			I[x,y] = np.sum(np.multiply(k, region))

	f = unpadding(f, a)
	I = unpadding(I, a)	

	I = normalize(I)
	r = c * I + f
	return normalize(r)

#Apply the Vignette Filter
def method3(m, SigmaRow, SigmaCol):
	M,N = m.shape
	kRow = np.zeros((1,M)) #kernel for the Row
	kCol = np.zeros((1,N)) #kernel for the Column
	#Row
	#differing if the size of Row is even or not
	if M % 2 == 0: 
		a = int((M/2)) -1
	else:
		a = int(M/2)
	for x in range(-a, int(M/2)):
		kRow[0][x+a] = G(x, SigmaRow)

	#Col
	#differing if the size of Col is even or not
	if N % 2 == 0:
		b = int((N/2)) -1 
	else:
		b = int(N/2)
	for x in range(-b, int(N/2)):
		kCol[0][x+b] = G(x, SigmaCol)

	filter_ = kRow.transpose() * kCol

	return normalize(filter_ * m)


def main():
	filename = str(input()).rstrip()
	#m = imageio.imread("imgs/" + filename) #to test on my computer
	m = imageio.imread(filename)
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
		row = float(input())
		col = float(input())
		output_img = method3(m, row, col)

	print("{:.4f}".format(Error(m,output_img)))

	output_img = np.asarray(output_img, dtype="uint8")

	if save == 1:
		imageio.imwrite('output_img.png', output_img)


if __name__ == "__main__":
	main()