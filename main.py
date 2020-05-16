# Basic libraries
import io
import os
import sys
import argparse
import numpy as np
from os import walk

import feature_extract as test
# Scikit learn stuff
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--articles_per_author", type=int, default = 50, help="Number of articles to use for each author. If an author has fewer number of articles, we ignore it.")
parser.add_argument("--authors_to_keep", type=int, default = 0, help="Number of authors to use for training and testing purposes")
parser.add_argument("--data_folder", type=str, default = "data", help="Folder where author data is kept. Each author should have a separate folder where each article should be a separate file inside the author's folder.")


args = parser.parse_args()
ARTICLES_PER_AUTHOR = args.articles_per_author
AUTHORS_TO_KEEP = args.authors_to_keep
DATA_FOLDER = args.data_folder

def calculateTop5Accuracy(labels, predictionProbs):
	"""
	Takes as input labels and prediction probabilities and calculates the top-5 accuracy of the model
	"""
	acc = []
	for i in range(0, len(predictionProbs)):
		predProbs = predictionProbs[i]
		predProbsIndices = np.argsort(-predProbs)[:5]
		if labels[i] in predProbsIndices:
			acc.append(1)
		else:
			acc.append(0)

	return round(((acc.count(1) * 100) / len(acc)), 2)


# Load raw data from the folder
print("Data loading...")
folders = []
vector=[]
for(_,dirs,_) in walk(DATA_FOLDER):
	folders.extend(dirs)

authorArticles = []
labels = []
authorId = 0
for author in folders:
	authorFiles = []
	for(_,_,f) in walk(DATA_FOLDER + "/" + author):
		authorFiles.extend(f)

	if len(authorFiles) < ARTICLES_PER_AUTHOR:
		continue

	authorFiles = authorFiles[:ARTICLES_PER_AUTHOR]
	print("Loading %d files from %s" % (len(authorFiles), author))
	temp_vector=[]
	for file in authorFiles:
		data = open(DATA_FOLDER + "/" + author + "/" + file, "r").readlines()
		data = ''.join(str(line) for line in data)
		temp_vector=test.FeatureExtration(data,15,4) 
		authorArticles.append(data)
		vector += temp_vector
		for K in range(len(temp_vector)):
			labels.append(authorId)

	# Stop when we have stored data for AUTHORS_TO_KEEP
	authorId = authorId + 1
	if authorId == AUTHORS_TO_KEEP:
		break


from sklearn.utils import shuffle
vector = np.array(vector)
labels = np.array(labels)

vector,labels = shuffle(vector,labels)


print("\nTraining and testing...")
# Train and get results
accuracies, precisions, recalls, fscores, top5accuracies = [], [], [], [], []
for i in range(10): # Train and test 10 different times and average the results
	# Split data into training and testing
	trainData, testData, trainLabels, testLabels = train_test_split(vector, labels, test_size=0.2)

	# Convert raw corpus into tfidf scores
	# vectorizer = TfidfVectorizer(min_df = 10)
	# vectorizer.fit(trainData)
	# trainData = vectorizer.transform(trainData).toarray()
	# testData = vectorizer.transform(testData).toarray()
	
	# Create a classifier instance
	classifier = RandomForestClassifier(n_estimators = 120)
	# classifier = KNeighborsClassifier(n_neighbors= 3)

	# Train classifier
	classifier.fit(trainData, trainLabels)

	# Get test predictions
	testPredictions = classifier.predict(testData)
	testPredictionsProbs = classifier.predict_proba(testData)
	testTopFiveAccuracy = calculateTop5Accuracy(testLabels, testPredictionsProbs)

	# Calculate metrics
	accuracy = round(accuracy_score(testLabels, testPredictions) * 100, 2)
	precision = round(precision_score(testLabels, testPredictions, average = 'macro') * 100, 2)
	recall = round(recall_score(testLabels, testPredictions, average = 'macro') * 100, 2)
	fscore = round(f1_score(testLabels, testPredictions, average = 'macro',labels=np.unique(testPredictions)) * 100, 2)
	confusionMatrix = confusion_matrix(testLabels, testPredictions)

	# Store metrics in lists
	accuracies.append(accuracy) 
	precisions.append(precision) 
	recalls.append(recall) 
	fscores.append(fscore) 
	top5accuracies.append(testTopFiveAccuracy)
	

print("Accuracy: ",round(max(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores)),2))