import random
import math

eps = 0.5 

def flipcoin():
	return 1 if random.random() > 0.5 else 0

def flipcoinNtimes(numberFlips):
	return [flipcoin() for i in range(numberFlips)]

def flipNcoinsMtimes(numberCoins, numberFlips):
	return [flipcoinNtimes(numberFlips) for i in range(numberCoins)]

def getMinHead(dataFromCoinFlips):
	mini = dataFromCoinFlips[0]
	for i in range(1, len(dataFromCoinFlips)):
		#print(dataFromCoinFlips[i])
		if(sum(mini) > sum(dataFromCoinFlips[i])):
			mini = dataFromCoinFlips[i]
	#print("this is the minimum ", mini, sum(mini))
	return mini

def getNuFromCoinFlipData(numberCoins, numberFlips):
	dataFromCoinFlips = flipNcoinsMtimes(numberCoins, numberFlips)
	cfirst = dataFromCoinFlips[0]
	crandom = dataFromCoinFlips[random.randint(0,numberCoins-1)]
	cmin = getMinHead(dataFromCoinFlips)
	return (sum(cfirst)/numberFlips, sum(crandom)/numberFlips, sum(cmin)/numberFlips)

def doExperiment(experiments, numberCoins, numberFlips):
	first, random, mini = zip(*[getNuFromCoinFlipData(numberCoins, numberFlips) for i in range(experiments)])
	print("mini ", mini)
	#eps = 0.1
	firststeps = [i for i in first if abs(i-0.5) >= eps]
	randomsteps = [i for i in random if abs(i-0.5) >= eps]
	minsteps = [i for i in mini if abs(i-0.5) >= eps]
	print(len(firststeps), " first steps ", firststeps)
	print(len(randomsteps), " random steps ", randomsteps)
	print(len(minsteps), " mini steps ", minsteps)
	return (len(firststeps)/experiments, len(randomsteps)/experiments, len(minsteps)/experiments)

if __name__ == '__main__':
	print("for eps ", eps, "value is", doExperiment(100000, 1000, 10), " RHS is = ", 2 * math.exp(-2*eps*eps*10))
	#print(2 * math.exp(-2*0.005*0.005*10))
