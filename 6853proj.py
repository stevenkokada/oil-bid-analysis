from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import csv
import math

np.random.seed(1234)

def importAuctionData():
	with open('oilBid.csv', 'r') as csvfile:
		filereader = csv.DictReader(csvfile)
		auctionDict = dict()
		for row in filereader:
			tractNo = int(row['TractNumber'])
			bid = int(row['Bid'].strip().replace(',',''))
			nBidders = int(row['nBidsOnTract'])+1
			if nBidders > 1:
			#only consider auctions where bidder was only one
				if tractNo in auctionDict:
					auctionDict[tractNo][1].append(bid)

				else:
					auctionDict[tractNo] = (nBidders, [bid])


	return auctionDict


def computeReduction(n, sigma):
	#checked against computations in paper
	#filter for n > 1

	integrand1 = lambda x: x*stats.norm.cdf(x, scale = sigma)**(n-2)*stats.norm.pdf(x, scale = sigma)**2
	integrand2 = lambda x: stats.norm.cdf(x, scale = sigma)**(n-2)*stats.norm.pdf(x, scale = sigma)**2
	num = 1/(n*n - n) + integrate.quad(integrand1, -np.inf, np.inf)[0]
	den = integrate.quad(integrand2, -np.inf, np.inf)[0]

	return num/den


def computeValue(auction):
	n = auction[0]
	bidVec = np.array(auction[1])/1000
	mu, std = stats.norm.fit(bidVec)
	# print(std)
	return (mu + computeReduction(n, std)/1000)*1000



def computePCurse(winBids, values):
	return (np.array(values) - np.array(winBids))/np.array(values)

def computeAvgPCurse(curses):
	return np.median(curses)

def computeAvgVal(values):
	return np.median(values)

def computeAvgBid(winBids):
	return np.median(winBids)

def computeStdVal(values):
	return np.std(values)

def computeStdBid(winBids):
	return np.std(winBids)


# helper function from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list

def reject_outliers(data, m = 2.):
	data = np.array(data)
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s<m]

data = importAuctionData()
winBids = []
values = []
avgBidders = 0

# print(len(data))
# counter = 0
# for tract in data:
# 	print(counter)
# 	avgBidders += data[tract][0]
# 	winBids.append(data[tract][1][0])
# 	values.append(computeValue(data[tract]))
# 	counter += 1

# avgBidders = avgBidders/counter

# curses = np.array(values) - np.array(winBids)
# pcurses = computePCurse(winBids, values)
# ratios = np.array(values)/np.array(winBids)

# values = reject_outliers(values)
# winBids = reject_outliers(winBids)
# curses = reject_outliers(curses)
# pcurses = reject_outliers(pcurses)
# ratios = reject_outliers(ratios)

# print("avgBidders:", avgBidders)
# print("avgPCurse:", computeAvgPCurse(pcurses))
# print("avgRatio:", np.mean(ratios))
# print("avgCurse:", np.median(curses))
# print("avgVal:", computeAvgVal(values))
# print("avgBid:", computeAvgBid(winBids))
# print("stdVal:", computeStdVal(values))
# print("stdBid:", computeStdBid(winBids))


# plots bid vector distribution and also gaussian fit
throwaway1, bins, throwaway2 = plt.hist(data[1653][1], bins = 20)
mu, std = stats.norm.fit(data[1653][1])

binwidth = bins[1] - bins[0]

plt.plot([x for x in bins], [len(data[1653][1])*binwidth*stats.norm.pdf(x, loc = mu, scale = std) for x in bins])
plt.xlabel('Bid Value (USD)')
plt.ylabel('Occurrences')

plt.show()
plt.figure()



# reductionF5 = lambda x: computeReduction(5, x)
# reductionF10 = lambda x: computeReduction(10, x)
# reductionF15 = lambda x: computeReduction(15, x)

# reductionN = lambda x: computeReduction(x, 1000)

# # plots reduction terms for varying sigmas and varying N

# f5, = plt.plot([10**x for x in range(4)], [reductionF5(10**x) for x in range(4)], label = "N = 5")
# f10, = plt.plot([10**x for x in range(4)], [reductionF10(10**x) for x in range(4)], label = "N = 10")
# f15, = plt.plot([10**x for x in range(4)], [reductionF15(10**x) for x in range(4)], label = "N = 15")
# plt.legend(handles = [f5, f10, f15])
# plt.show()
# plt.figure()

# # # plots reduction term for varying N with fixed typical sigma

# plt.scatter([x for x in range(2,20)], [reductionN(x) for x in range(2,20)])
# plt.xlabel('Number of Bidders, sigma = 1000')
# plt.ylabel('BNE Reduction (USD)')
# plt.show()
# plt.figure()

# # plots difference between values and winBids;


# plt.scatter([x for x in range(len(curses))], curses)
# plt.xlabel('Index')
# plt.ylabel('Value Estimate - Winning Bid (USD)')
# plt.show()
# plt.figure()

# # # plots difference in y=x relationship
# plt.scatter([x for x in range(len(ratios))], ratios)
# plt.xlabel('Index')
# plt.ylabel('Value Estimate / Winning Bid')
# plt.show()
# plt.figure()

# plots histogram of bids, valuations

# values = reject_outliers(values)
# winBids = reject_outliers(winBids)
# plt.hist(values, bins = 30)
# plt.xlabel('Value Estimate (USD)')
# plt.ylabel('Occurrences')
# plt.show()
# plt.figure()

# plt.hist(winBids, bins = 30)
# plt.xlabel('Winning Bid (USD)')
# plt.ylabel('Occurrences')
# plt.show()
