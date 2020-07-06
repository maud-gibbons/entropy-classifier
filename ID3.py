import numpy as np
import pandas as pd
import scipy
from collections import Counter

#function to calculate the entropy pf a set of examples
def entropy(examples, threshold):
	num_attribute = 0
	num_non_attribute = 0

	for i in range(len(examples)):
		if(examples[i][1] == threshold[1]):
			num_attribute = num_attribute+1
		else:
			num_non_attribute = num_non_attribute+1

	pA =  float(num_attribute)/float(len(examples))
	pB = float(num_non_attribute)/float(len(examples))

	#The only time this is true is when the child classes are perfectly split by the threshold and entropy for that part will be zero anyway
	#Avoids calculation with undefined log(0)
	if(pA == 0):
		pA = 1.0
	if(pB == 0):
		pB = 1.0

	entropy = -(pA)*np.log2(pA) -(pB)*np.log2(pB)
	return entropy

#function to calculate the information gain pf a set of examples
def informationGain(examples, examples_right, examples_left, threshold):
	entropy_before = entropy(examples,threshold)
	entropy_right = entropy(examples_right,threshold)
	entropy_left = entropy(examples_left,threshold)
	# print
	# print entropy_before
	# print entropy_right
	# print entropy_left

	entropy_after = ( float(len(examples_left))/ float(len(examples)))*entropy_left + ( float(len(examples_right))/ float(len(examples)))*entropy_right
	#print entropy_after
	information_gain = entropy_before - entropy_after
	#print information_gain
	return information_gain

#determine the candidate thresholds for a feature
def cal_candidate_thresholds(feature):
	points = []
	for i in range(len(feature)-1):
		t1 = feature[i]
		t2 = feature[i+1]
		if(t1[0] != t2[0]and t1[1] != t2[1]):
			points.append(t1)
			points.append(t2)

	thresholds = []
	i = 0
	while (i < len(points)):
		point =  round((float(points[i][0]) + float(points[i+1][0]))/2,2)
		threashold = [point, points[i][1], "Not"+points[i][1]]
		thresholds.append(threashold)
		i = i+2
	return thresholds

#determine the best threshold for a feature
def cal_best_threshold(feature):
	candaditeThresholds = cal_candidate_thresholds(feature)
	best_threshold = []
	best_information_gain = 0
	#Calculate theh information gain for each threshold
	for a in range(len(candaditeThresholds)):
		left = []
		right = []
		#apply the threshold
		for i in range(len(feature)):
			if(feature[i][0] < candaditeThresholds[a][0]):
				left.append(feature[i])
			else:
				right.append(feature[i])

		if(len(right)>0 or len(left)>0):
			information_gain = informationGain(feature, right, left, candaditeThresholds[a])
			if information_gain > best_information_gain:
				best_information_gain = information_gain
				best_threshold = candaditeThresholds[a]
	return [best_threshold,best_information_gain]

#determine the feature that best seperates the data
def cal_best_feature(feature):
	candidate_feature = []
	best_feature = []
	for x in range(len(feature)):
		feature[x].sort()
		best_threshold = cal_best_threshold(feature[x])
		candidate_feature.append([x,best_threshold[0],best_threshold[1]])

	best = 0.0
	for x in range(len(candidate_feature)):
		if candidate_feature[x][2] >= best:
			best = candidate_feature[x][2]
			best_feature = candidate_feature[x]
	return best_feature

def extract(data):
	#extract the features
	features = [0 for i in range(len(data[0])-1)]
	class_pos = len(data[0])-1

	for i in range(len(features)):
		features[i] = [0 for j in range(len(data))]

	for i in range(len(data)):
		for j in range(len(features)):
			features[j][i] = [ data[i][j] , data[i][class_pos] ]
	return features

def build_classifer(examples,attributes):
	if examples == []:
		return "Empty"

	if len(attributes) == 0:
			return Counter(attributes)

	same = True
	majority = False
	class_pos = len(examples[0])-1
	for i in range(len(examples)-1):
		if examples[i][class_pos] != examples[i+1][class_pos]:
			same = False

	if same == True:
		print "all ",examples[0][class_pos]
		return examples[0][class_pos]

	else:
		root = cal_best_feature(attributes)
		print root
		left = []
		right = []
		for i in range(len(examples)):
			if(examples[i][root[0]] < root[1][0]):
				left.append(examples[i])
			else:
				right.append(examples[i])

		build_classifer(list(left),list(extract(left)))
		build_classifer(list(right),list(extract(right)))
		return "Done"


f = pd.read_csv('owls.csv')
arr = np.array(f)
owls = []
for item in arr:
	owls.append(list(item))

print build_classifer(list(owls),list(extract(owls)))
