
if __name__ == '__main__':
	degree = 10
	count = 0
	k = []
	for i in range(degree+1):
		#print("x%s" %i)
		for j in range((degree+1)-i):
			print('x%sy%s' %(i,j))
			k.append('x%sy%s' %(i,j))
			count+=1
	print(count)
	print(k)

		