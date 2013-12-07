import random

def flipcoin():
	return 1 if random.random() > 0.5 else 0

def flipcoinNtimes(numberFlips):
	return [flipcoin() for i in range(numberFlips)]

def flipNcoinsMtimes(numberCoins, numberFlips):
	return [flipcoinNtimes(numberFlips) for i in range(numberCoins)]

def getMinHead(dataFromCoinFlips):
	mini = dataFromCoinFlips[0]
	for i in range(1, len(dataFromCoinFlips)):
		if(sum(mini) > sum(dataFromCoinFlips[i])):
			mini = dataFromCoinFlips[i]
	return mini

def getNuFromCoinFlipData(numberCoins, numberFlips):
	dataFromCoinFlips = flipNcoinsMtimes(numberCoins, numberFlips)
	cfirst = dataFromCoinFlips[0]
	crandom = dataFromCoinFlips[random.randint(0,numberCoins-1)]
	cmin = getMinHead(dataFromCoinFlips)
	return (sum(cfirst)/numberFlips, sum(crandom)/numberFlips, sum(cmin)/numberFlips)

def doExperiment(experiments, numberCoins, numberFlips):
	first, random, mini = zip(*[getNuFromCoinFlipData(numberCoins, numberFlips) for i in range(experiments)])
	return (sum(first)/experiments, sum(random)/experiments, sum(mini)/experiments)

if __name__ == '__main__':
	(firstnu, randomnu, minnu) = doExperiment(100000, 1000, 10)
	mu = 0.5
	
