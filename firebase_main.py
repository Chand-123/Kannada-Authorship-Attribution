# Basic libraries
import io
import os
import sys
import argparse
import numpy as np
from os import walk
from sklearn import svm
import feature_extract as test
# Scikit learn stuff
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from firebase import firebase
firebase_db = firebase.FirebaseApplication('https://hackathon-ab821.firebaseio.com/', None)
import pyrebase

config = {
  "apiKey": "AIzaSyB_6OR0rAtv3xgCzQC45A0mjdzTW_KF2cw",
  "authDomain": "hackathon-ab821.firebaseapp.com",
  "databaseURL": "https://hackathon-ab821.firebaseio.com",
  "projectId": "hackathon-ab821",
  "storageBucket": "hackathon-ab821.appspot.com",
  "messagingSenderId": "407886884054",
  "appId": "1:407886884054:web:9b0c1709e25124bb"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
Train_status = {
	"status" : "null",
	"text": "null"	
	}
firebase_db.put('','Train',Train_status)
msg = {
	"status" : "null"	
	}
firebase_db.put('','msg',msg)
test_ = {
    "author" : "null",
    "Prediction array": "null",	
    "Prediction Probability": "null",	
	}
firebase_db.put('','Test',test_)


def stream_handler(message):
	global best_classifier
    # print(message["event"]) # put
    # print(message["path"]) # /-K7yGTTEp7O549EzTYtI
	# print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
	print(message)
	status = message["data"]['status']# {'title': 'Pyrebase', "body": "etc..."}
	text = message["data"]['text']# {'title': 'Pyrebase', "body": "etc..."}
	print(status)
	if (status == 'start'):
		best_classifier,best_accuracy = Train()
		Train_status = {
			"status" : "trained"+'_'+str(best_accuracy),
			"text": "null"	
			}
		firebase_db.put('','Train',Train_status)
	
	
	if (status.__contains__('trained') and text != "null"):
		Test(best_classifier)
	

print('Initialising...')
my_stream = db.child("Train").stream(stream_handler)


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


# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--articles_per_author", type=int, default = 12, help="Number of articles to use for each author. If an author has fewer number of articles, we ignore it.")
parser.add_argument("--authors_to_keep", type=int, default = 3, help="Number of authors to use for training and testing purposes")
parser.add_argument("--data_folder", type=str, default = "data", help="Folder where author data is kept. Each author should have a separate folder where each article should be a separate file inside the author's folder.")
parser.add_argument("--model", type=str, default = "RandomForest", help="Training model")
parser.add_argument("--doc", type=str, default = "test.txt", help="Testing input")
parser.add_argument("--id", type=int, default = 5, help="id")


args = parser.parse_args()
ARTICLES_PER_AUTHOR = args.articles_per_author
AUTHORS_TO_KEEP = args.authors_to_keep
DATA_FOLDER = args.data_folder
MODEL = args.model
DOC = args.doc
ID = args.id


def Train():
	# Load raw data from the folder
	print("Data loading...")
	msg_status = {
			"status" : "Data loading...",
				
			}
	firebase_db.put('','msg',msg_status)
	Train_status = {
			"status" : "training",
			"text": "null"	
			}
	firebase_db.put('','Train',Train_status)
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
		msg_status = {
			"status" : "Loading "+str(len(authorFiles)) +" files from "+str(author.replace('data_','')),
				
			}
		firebase_db.put('','msg',msg_status)
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
	print('ML model :',MODEL.upper(),'\n\n')
	# Train and get results
	accuracies, precisions, recalls, fscores, top5accuracies = [], [], [], [], []
	temp_accuracy = 0
	for i in range(10): # Train and test 10 different times and average the results
		# Split data into training and testing
		trainData, testData, trainLabels, testLabels = train_test_split(vector, labels, test_size=0.2)

		# Convert raw corpus into tfidf scores
		# vectorizer = TfidfVectorizer(min_df = 10)
		# vectorizer.fit(trainData)
		# trainData = vectorizer.transform(trainData).toarray()
		# testData = vectorizer.transform(testData).toarray()
		
		# Create a classifier instance
		if MODEL == 'RandomForest':
			classifier = RandomForestClassifier(n_estimators = 120)
		if MODEL == 'svm':
			classifier =  svm.SVC(kernel='linear',probability=True)
		if MODEL == 'NaiveBayes':
			classifier =  GaussianNB()
		if MODEL == 'knn':
			classifier = KNeighborsClassifier(n_neighbors= 3)
		

		# classifier =  svm.SVC(kernel='linear',probability=True)
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

		# Store metrics in lists
		if(temp_accuracy< accuracy):
			temp_classifier=classifier
			temp_accuracy=accuracy
			# print('best score,', temp_accuracy)
		accuracies.append(accuracy)
		# print('Accuracy:', accuracy)
		precisions.append(precision) 
		recalls.append(recall) 
		fscores.append(fscore) 
		top5accuracies.append(testTopFiveAccuracy)
	
	best_accuracy = round(max(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores)),2)
	print("Accuracy: ",best_accuracy)
	return temp_classifier,best_accuracy


def Test(classifier):
	temp_vector = []
	test_vector = []
	test_labels = []
	data = firebase_db.get('/Train',None)['text']
	data = ''.join(str(line) for line in data)
	# print(data)
	temp_vector=test.FeatureExtration(data,15,4)
	test_vector += temp_vector
	# for K in range(len(temp_vector)):
	# 	test_labels.append(ID)
	testPredictions = classifier.predict(test_vector)
	testPredictionsProbs = classifier.predict_proba(test_vector)
	print(testPredictions)
	arr = testPredictions.tolist()
	if(max(arr,key=arr.count) == 0):
		name = 'Somashekar'
	if(max(arr,key=arr.count) == 1):
		name = 'Hrudayashiva'
	if(max(arr,key=arr.count) == 2):
		name = 'Ravi belegere'
	
	Test_status = {
			"author" : name+'_author'	
			}
	
	# Test_status = {
	# 		"author" : name+'_author',
	# 		"Prediction array": str(arr),	
	# 		"Prediction Probability": str(testPredictionsProbs),	
	# 		}
	firebase_db.put('','Test',Test_status)
	print('Author Name: ', name)
	print(testPredictionsProbs)
	