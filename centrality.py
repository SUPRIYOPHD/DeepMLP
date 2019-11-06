import json
from operator import itemgetter
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

import math as mt
import csv
from sparsesvd import sparsesvd

def harmonic_sum(n):
	sum = 0
	for i in range(1,n+1):
		sum += (1.0/i)
	return sum

def computeSVD(urm, K):
	U, s, Vt = sparsesvd(urm.tocsc(), K)
	# print "************************"
	# print urm.todense()
	# print np.dot(U.T,np.dot(np.diag(s),Vt))

	# print urm.shape[0]
	# print urm.shape[1]
	# dim = (urm.shape[0], urm.shape[1])
	S=np.diag(s)
	# S = np.zeros(dim, dtype=np.float32)
	# for i in range(0, min(urm.shape[0], urm.shape[1])):
	# 	S[i,i] = mt.sqrt(s[i])

		# U = csr_matrix(U.T, dtype=np.float32)
		# S = csr_matrix(S, dtype=np.float32)
		# Vt = csr_matrix(Vt, dtype=np.float32)

	return U, S, Vt	

data = []
with open('./trunc1L.json','r') as f:
	for line in f:
		data.append(json.loads(line))

# finding the helpfulness score for each review
for l in data:
	if l["helpful"][1]!=0 :
		l["helpfulness_score"] = l["helpful"][0]**2.0/l["helpful"][1]
	else :
		l["helpfulness_score"] = 0

# finding the rank of each review
sorted_helpful = sorted(data, key=itemgetter('helpfulness_score'), reverse=True) 

productMap = {}

for l in sorted_helpful:
	if l["asin"] in productMap:

		if productMap[l["asin"]][1] == l["helpfulness_score"]:
			l["rank"]=productMap[l["asin"]][0]
			productMap[l["asin"]][2] += 1

		else:
			productMap[l["asin"]][2] += 1
			l["rank"]=productMap[l["asin"]][2]
			productMap[l["asin"]][0] = productMap[l["asin"]][2]
			productMap[l["asin"]][1] = l["helpfulness_score"]

	else:
		productMap[l["asin"]] = [1,l["helpfulness_score"],1]
		l["rank"]=productMap[l["asin"]][0]

# finding the position of each review
sorted_time = sorted(sorted_helpful, key=itemgetter('unixReviewTime'))

productMap = {}

for l in sorted_time:
	if l["asin"] in productMap:
		productMap[l["asin"]] += 1
		l["count"]=productMap[l["asin"]]

	else:
		productMap[l["asin"]] = 1
		l["count"]=productMap[l["asin"]]

# finding the degree of each review
for l in sorted_time:
	l["degree_topmost"]=(1.0/(l["rank"]**2))*(productMap[l["asin"]]-l["count"])
	l["degree_mostrecent"]=harmonic_sum(productMap[l["asin"]]-l["count"])
	l["degree"]=l["degree_topmost"]+l["degree_mostrecent"]
	
# f = open('centrality.txt','w');
sorted_itemwise = sorted(sorted_helpful, key=itemgetter('asin'))

# for l in sorted_itemwise:
# 	print "reviewerID : %s, asin : %s, time : %s, rating : %f, helpfulness_score : %f, centrality : %f, rank : %d" %(l["reviewerID"],l["asin"],l["reviewTime"],l["overall"],l["helpfulness_score"],l["degree"],l["rank"])
	# f.write("reviewerID : %s, asin : %s, time : %s, rating : %f, helpfulness_score : %f, centrality : %f\n" %(l["reviewerID"],l["asin"],l["reviewTime"],l["overall"],l["helpfulness_score"],l["degree"]))

countU=0
countP=0
uIndex={}
pIndex={}

for l in sorted_itemwise:
	if l["reviewerID"] not in uIndex:
		uIndex[l["reviewerID"]]=countU
		countU+=1

	if l["asin"] not in pIndex:
		pIndex[l["asin"]]=countP
		countP+=1

MAX_UID = countU
MAX_PID = countP

urmRating = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
urmHelpful = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)
urmCentrality = np.zeros(shape=(MAX_UID,MAX_PID), dtype=np.float32)

for l in sorted_itemwise:
	urmRating[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["overall"])
	urmHelpful[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["helpfulness_score"])
	urmCentrality[uIndex[l["reviewerID"]], pIndex[l["asin"]]] = float(l["degree"])

# select a user,product to predict. Implement a function to find users who have purchased multiple products.
# select any one of the product and remove it.

urmCsrRating = csr_matrix(urmRating, dtype=np.float32)
urmCsrHelpful = csr_matrix(urmHelpful, dtype=np.float32)
urmCsrCentrality = csr_matrix(urmCentrality, dtype=np.float32)

# pass urmCsrRating to a matrix. convert it into 
U, s1, Vt = computeSVD(urmCsrRating, 5)
X, s2, Yt = computeSVD(urmCsrHelpful, 5)
S, s3, Tt = computeSVD(urmCsrCentrality, 5)

# print "U shape : ",U.shape[0]," ",U.shape[1]
# print "Vt shape : ",Vt.shape[0]," ",Vt.shape[1]

eps = 0.005

U=U.T
X=X.T
S=S.T

# Predict rating using the rating matrix
for iterations in range(10):
	predictedRating = (U.dot(s1.dot(Vt)))

	psi = 0
	for i in range(len(urmRating)):
		for j in range(len(urmRating[i])):
			err = urmRating[i][j] - predictedRating[i,j]

			psi += err**2

			U[i] += eps * err * Vt.T[j]
			Vt.T[j] += eps * err * U[i]

	# print "Psi",iterations," value : ",psi

predictedRating = (U.dot(s1.dot(Vt)))

print ("predictedRating.min : ",predictedRating.min())
print ("predictedRating.max : ",predictedRating.max())

minimum = predictedRating.min()
maximum = predictedRating.max()

for i in range(len(predictedRating)):
	for j in range(len(predictedRating[i])):
		predictedRating[i][j] = 1 + (predictedRating[i][j] - minimum) * (4/(maximum - minimum))

# Predict rating using the helpfulness matrix
for iterations in range(10):
	predictedRating1 = (X.dot(s2.dot(Yt)))

	psi = 0
	for i in range(len(urmHelpful)):
		for j in range(len(urmHelpful[i])):
			err = urmHelpful[i][j] - predictedRating1[i,j]

			psi += err**2

			X[i] += eps * err * Yt.T[j]
			Yt.T[j] += eps * err * X[i]

	# print "Psi",iterations," value : ",psi

predictedRating1 = (X.dot(s2.dot(Vt)))

print ("predictedRating1.min : ",predictedRating1.min())
print ("predictedRating1.max : ",predictedRating1.max())

minimum = predictedRating1.min()
maximum = predictedRating1.max()

for i in range(len(predictedRating1)):
	for j in range(len(predictedRating1[i])):
		predictedRating1[i][j] = 1 + (predictedRating1[i][j] - minimum) * (4/(maximum - minimum))

# Predict rating using the centrality matrix
for iterations in range(10):
	predictedRating2 = (S.dot(s3.dot(Tt)))

	psi = 0
	for i in range(len(urmCentrality)):
		for j in range(len(urmCentrality[i])):
			err = urmCentrality[i][j] - predictedRating2[i,j]

			psi += err**2

			S[i] += eps * err * Tt.T[j]
			Tt.T[j] += eps * err * S[i]

	# print "Psi",iterations," value : ",psi

predictedRating2 = (S.dot(s3.dot(Vt)))

print ("predictedRating2.min : ",predictedRating2.min())
print ("predictedRating2.max : ",predictedRating2.max())

minimum = predictedRating2.min()
maximum = predictedRating2.max()

for i in range(len(predictedRating2)):
	for j in range(len(predictedRating2[i])):
		predictedRating2[i][j] = 1 + (predictedRating2[i][j] - minimum) * (4/(maximum - minimum))

print ("The rating matrix is \n",urmRating)
aggregatedRating = (predictedRating+predictedRating1+predictedRating2)/3.0
# print "The predicted rating(rating) is \n",predictedRating
# print "The predicted rating(1) is \n",predictedRating1
# print "The predicted rating(2) is \n",predictedRating2


print ("The predicted rating is \n",aggregatedRating)

# calculate the mean absolute error
num_purchases = 0
absError = 0
for i in range(len(urmRating)):
	for j in range(len(urmRating[i])):
		if urmRating[i][j] != 0:
			absError += abs(aggregatedRating[i][j] - urmRating[i][j])
			num_purchases += 1

meanAbsError = absError / num_purchases
print ("Mean Absolute Error : ",meanAbsError)

f.close()