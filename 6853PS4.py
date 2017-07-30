import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def computeBNE(value):
	"""takes in value and returns corresponding bid at Bayesian Nash Eq"""

	# def CDFMax(x):
	# 	return stats.uniform.cdf(x)**(n-1)


	# integral = integrate.quad(CDFMax,0,value)[0]

	# bid = value - (1/stats.uniform.cdf(value))**(n-1)*integral

	return (3/4)*value


def computeECDF(bids, x):
	"""calculates empirical cumulative distribution function based on list of values/bids and returns its value at x"""
	bids = sorted(bids)
	norm = len(bids)
	counter = 0

	for i in range(len(bids)):
		current = bids[i]
		if current > x:
			return counter/norm

		counter += 1

	return counter/norm


def uniformKernel(u):
	if abs(u) > 1:
		return 0

	return 0.5

def epanechnikovKernel(u):
	if abs(u) > 1:
		return 0

	return 0.75*(1-u*u)


def computeEPDF(bids, kernel, bandwidth, x):
	"""calculates empirical probability density function based on list of values/bids, kernel, and bandwidth, and returns its value at x"""

	norm = len(bids)
	total = 0
	for i in range(len(bids)):
		total += (1/bandwidth)*kernel((bids[i]-x)/bandwidth)

	return total/norm


def invertApproxBid(G, g, bid):
	"""returns estimated value from cdf + pdf + calculated bid"""
	return bid + G(bid)/(n-1)/g(bid)


def inversionError(estimatedValues, samples):
	"""calculates sum of L1 norm difference between estimatedValues and actual values"""
	diff = [abs(estimatedValues[i] - samples[i]) for i in range(len(estimatedValues))]
	return sum(diff)



# part a)
# valuationNo = 1000
# part f)
valuationNo = 100000
kernels = [uniformKernel, epanechnikovKernel]

samples = stats.uniform.rvs(size=valuationNo)

# part b)
n = 4


samples = np.array(samples)
bids = computeBNE(samples)

# part c)
uniformEPDF = []
epanechnikovEPDF = []

G = lambda x: computeECDF(bids, x)


bidsToPlot = sorted(bids)
plt.plot(bidsToPlot, [G(x) for x in bidsToPlot])
plt.plot(bidsToPlot, [4/3 * x for x in bidsToPlot], '--')
plt.show()



plt.figure()
plt.plot(bidsToPlot, [computeEPDF(bids, uniformKernel, 0.5, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, uniformKernel, 0.1, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, uniformKernel, 0.05, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, uniformKernel, 0.01, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [4/3 for x in bidsToPlot], '--')
plt.show()



plt.figure()
plt.plot(bidsToPlot, [computeEPDF(bids, epanechnikovKernel, 0.5, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, epanechnikovKernel, 0.1, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, epanechnikovKernel, 0.05, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [computeEPDF(bids, epanechnikovKernel, 0.01, x) for x in bidsToPlot])
plt.plot(bidsToPlot, [4/3 for x in bidsToPlot], '--')
plt.show()



# part d)

neworder = list(zip(samples, bids))
neworder.sort(key = lambda x: x[0])

samplesOrdered = [blah[0] for blah in neworder]
bidsOrdered = [blah[1] for blah in neworder]


inversionResults = []
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.5, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.1, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.05, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.01, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.5, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.1, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.05, x), bid) for bid in bidsOrdered])
inversionResults.append([invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.01, x), bid) for bid in bidsOrdered])

errorResults = [inversionError(samples, inversionResults[i]) for i in range(len(inversionResults))]
print(errorResults)
print(np.argmin(errorResults))







plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.5, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.1, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.05, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, uniformKernel, 0.01, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, samplesOrdered)



plt.show()


	
plt.figure()

plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.5, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.1, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.05, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, [invertApproxBid(G, lambda x: computeEPDF(bidsOrdered, epanechnikovKernel, 0.01, x), bid) for bid in bidsOrdered])
plt.plot(samplesOrdered, samplesOrdered)

plt.show()



# part e)

vHats = inversionResults[np.argmin(errorResults)]
F = lambda x: computeECDF(vHats, x)

valuesToPlot = sorted(vHats)
plt.plot(valuesToPlot, [F(x) for x in valuesToPlot])
plt.plot(valuesToPlot, [min(x,1) for x in valuesToPlot], '--')
plt.show()



plt.figure()
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, uniformKernel, 0.5, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, uniformKernel, 0.1, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, uniformKernel, 0.05, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, uniformKernel, 0.01, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [1 for x in valuesToPlot], '--')
plt.show()



plt.figure()
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, epanechnikovKernel, 0.5, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, epanechnikovKernel, 0.1, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, epanechnikovKernel, 0.05, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [computeEPDF(valuesToPlot, epanechnikovKernel, 0.01, x) for x in valuesToPlot])
plt.plot(valuesToPlot, [1 for x in valuesToPlot], '--')
plt.show()



