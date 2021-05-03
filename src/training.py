import requests
import time

import numpy as np
import pymongo

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from champion_map import internal_champion_map, n_champs
from config import LOL_DB_NAME, MATCH_COL_NAME
from training_config import *

from HIDDEN_CONFIG import MONGODB_CONN_STR

# Convert document to a one hot encoding of champions on each team
def conv_document_to_examples(document):
	team1_OHE = np.zeros(n_champs)
	team2_OHE = np.zeros(n_champs)
	for player in document['players']:
		champion = internal_champion_map[player['champion']]
		champ_OHE = np.zeros(n_champs)
		champ_OHE[champion] = 1
		if player["team"]:
			team2_OHE += champ_OHE
		else:
			team1_OHE += champ_OHE
		example1 = np.concatenate((example1, ohe))
		#example2 = np.concatenate((ohe, example2))
	example = np.concatenate((team1_OHE, team2_OHE))
	example = np.append(example, int(document['outcome']))
	return example

# Get n matches (all matches in database if n < 0)
def get_matches(matchCol, n = -1):
	cursor = matchCol.find({})
	matches = []
	for document in cursor:
		example = conv_document_to_examples(document)
		matches.append(example)
		if len(matches) % 1000 == 0:
			print("{} matches downloaded from database...\n".format(len(matches)))
		if n >= 0:
			if len(matches) >= n:
				return matches
	return matches

# Swap the teams data in an example
def dup_swap_x(x):
	return np.concatenate((x[n_champs:],x[:n_champs]))

# Plot the history of a metric for training and testing
def plot_metric_history(model, history, metric_name):
	metric = history.history[metric_name]
	val_metric = history.history['val_' + metric_name]

	e = range(1, NUM_EPOCHS + 1)

	plt.figure()
	plt.plot(e, metric, 'b', label='Train ' + metric_name)
	plt.plot(e, val_metric, 'r', label='Testing ' + metric_name)
	plt.xlabel('Epoch number')
	plt.ylabel(metric_name)
	plt.title('Comparing training and testing ' + metric_name + ' for ' + model.name)
	plt.legend()
	plt.savefig(metric_name + '.png')

# Find the best epoch in the model for test set loss
def best_epoch(history):
	me = np.argmin(history.history['val_loss']) + 1
	print("Minimum test loss reached in epoch {} with validation accuracy {}".format(me, history.history['val_accuracy'][me-1]))
	return me

# Boost the training data set size by making the model team side invariant
def boost_training_data(X_train, y_train):
	print("Boosting training data to make the model team side invariant...")
	newX_train = []
	newy_train = []
	for i in range(len(y_train)):
		X = X_train[i]
		y = y_train[i]
		newX_train.append(X)
		newX_train.append(dup_swap_x(X))
		newy_train.append(y)
		newy_train.append(float(not bool(y)))
	newX_train = np.array(newX_train)
	newy_train = np.array(newy_train)
	return newX_train, newy_train

def get_training_data():
	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	match_col = loldb[MATCH_COL_NAME]

	matches = np.array(get_matches(match_col, n_matches))
	X = matches[:,:-1]
	y = matches[:,-1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PORTION)
	
	if DO_MATCH_DUP:
		X_train, y_train = boost_training_data(X_train, y_train)

	return X_train, X_test, y_train, y_test

# Create a sequential model to train on
def create_training_model():
	model = keras.Sequential()
	model.add(keras.layers.Dense(hidden_nodes[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_champs*2,)))
	for hn in hidden_nodes[1:]:
		model.add(keras.layers.Dense(hn, activation='relu', kernel_initializer='he_normal'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	opt = keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

	return model

# Train the model on training data using test data as validation
def train_model(model, X_train, X_test, y_train, y_test):
	if BATCH_SIZE < 0:
		BATCH_SIZE = len(y_train)

	history = model.fit(
		X_train, 
		y_train, 
		epochs = NUM_EPOCHS, 
		batch_size = BATCH_SIZE, 
		validation_data=(X_test,y_test),
		verbose = 1)

	loss, acc = model.evaluate(X_test, y_test, verbose = 1)
	return loss, acc

# Returns the model's accuracy with most confident predictions using multiple confidence levels
def get_confidence_accuracies(model, X, y, confidenceLevels=[]):
	if not len(confidenceLevels):
		print("Error: Please provide a confidence level to get the accuracy of")
		return None

	# Sort model predictions by most confident
	yhats = model.predict(X)
	yhats_confidences = []
	for i, yhat in enumerate(yhats):
		yhats_confidences.append(((yhat-0.5)**2, yhat, y[i]))
	yhats_confidences.sort(key=lambda x:x[0],reverse=True)

	# Calculate the accuracy at each confidence level
	accuracies = []
	for confidenceLevel in confidenceLevels:
		correct = 0
		confident_yhats = yhats[:int(len(yhats) * confidenceLevel)]
		for i, yhat_confidence in enumerate(confident_yhats):
			if int(np.round(yhat_confidence[1])) == int(yhat_confidence[2]):
				correct += 1
		accuracies.append(correct / len(confident_yhats))
	return accuracies

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = get_training_data()

	model = create_training_model()

	loss, acc = train_model(model, X_train, X_test, y_train, y_test)

	plot_metric_history(model, history, 'loss')
	plot_metric_history(model, history, 'accuracy')
	
	best_epoch(history)

	print('Test accuracy: %.3f' % acc)

	confidenceLevels = [1/10, 1/100]
	conf_accs = get_confidence_accuracies(model, X_test, y_test, confidenceLevels=confidenceLevels)
	for i in range(len(confidenceLevels)):
		print("%.1f% most confident predictions accuracy: %.3f%" % (confidenceLevels[i]* 100, conf_accs[i]) )