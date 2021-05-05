import requests
import time
import os

import numpy as np
import pymongo

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from champion_map import internal_champion_map, n_champs, champion_name_map
from config import LOL_DB_NAME, MATCH_COL_NAME
import training_config

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
		if len(matches) % 10000 == 0:
			print("{:,} matches downloaded from database...\n".format(len(matches)), flush=True)
		if n >= 0:
			if len(matches) >= n:
				return matches
	print("{:,} matches downloaded from database".format(len(matches)), flush=True)
	return matches

# Swap the teams data in an example
def dup_swap_x(x):
	return np.concatenate((x[n_champs:],x[:n_champs]))

# Plot the history of a metric for training and testing
def plot_metric_history(model, history, metric_name):
	metric = history.history[metric_name]
	val_metric = history.history['val_' + metric_name]

	e = range(1, training_config.NUM_EPOCHS + 1)

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

	matches = np.array(get_matches(match_col, training_config.n_matches))
	X = matches[:,:-1]
	y = matches[:,-1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training_config.TEST_PORTION)
	
	if training_config.DO_MATCH_DUP:
		X_train, y_train = boost_training_data(X_train, y_train)

	return X_train, X_test, y_train, y_test

# Create a sequential model to train on
def create_training_model():
	model = keras.Sequential()
	model.add(keras.layers.Dense(training_config.hidden_nodes[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_champs*2,)))
	for hn in training_config.hidden_nodes[1:]:
		model.add(keras.layers.Dense(hn, activation='relu', kernel_initializer='he_normal'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	opt = keras.optimizers.Adam(learning_rate=training_config.learning_rate)
	model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

	return model

# Train the model on training data using test data as validation
def train_model(model, X_train, X_test, y_train, y_test):
	if training_config.BATCH_SIZE < 0:
		training_config.BATCH_SIZE = len(y_train)

	checkpoint_dir = os.path.dirname(training_config.CHECKPOINT_PATH)

	# Create a callback that saves the model's weights
	#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=training_config.CHECKPOINT_PATH,
	#                                                 save_weights_only=True,
	#                                                 save_freq = training_config.NUM_EPOCHS * training_config.BATCH_SIZE,
	#                                                 verbose=1)

	history = model.fit(
		X_train, 
		y_train, 
		epochs = training_config.NUM_EPOCHS, 
		batch_size = training_config.BATCH_SIZE, 
		validation_data=(X_test,y_test),
		#callbacks=[cp_callback],
		verbose = 1)

	model.save_weights(training_config.SAVE_PATH)

	loss, acc = model.evaluate(X_test, y_test, verbose = 1)
	return loss, acc, history

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
		confident_yhats = yhats_confidences[:int(len(yhats) * confidenceLevel)]
		for i, yhat_confidence in enumerate(confident_yhats):
			if int(np.round(yhat_confidence[1])) == int(yhat_confidence[2]):
				correct += 1
		accuracies.append(correct / len(confident_yhats))
	return accuracies

if __name__ == '__main__':

	model = create_training_model()
	if training_config.DO_LOAD:
		X_train, X_test, y_train, y_test = get_training_data()

	if training_config.DO_TRAIN:
		loss, acc, history = train_model(model, X_train, X_test, y_train, y_test)

		plot_metric_history(model, history, 'loss')
		plot_metric_history(model, history, 'accuracy')
		
		best_epoch(history)

		print('Test accuracy: %.3f' % acc)
	else:
		model.load_weights(training_config.SAVE_PATH)

	model.summary()

	if training_config.DO_LOAD:
		confidenceLevels = [1/10, 1/100]
		conf_accs = get_confidence_accuracies(model, X_test, y_test, confidenceLevels=confidenceLevels)
		for i in range(len(confidenceLevels)):
			print("{:.2f}% most confident predictions accuracy: {:.2f}".format(confidenceLevels[i]* 100, conf_accs[i]) )

	print("\nTesting example game...")
	exampleGame = {
		"players": [
			{
				"champion": champion_name_map["Urgot"], 
				"team": 0
			},
			{
				"champion": champion_name_map["Rammus"], 
				"team": 0
			},
			{
				"champion": champion_name_map["Zed"], 
				"team": 0
			},
			{
				"champion": champion_name_map["Caitlyn"], 
				"team": 0
			},
			{
				"champion": champion_name_map["Nami"], 
				"team": 0
			},
			{
				"champion": champion_name_map["Nasus"], 
				"team": 1
			},
			{
				"champion": champion_name_map["JarvanIV"], 
				"team": 1
			},
			{
				"champion": champion_name_map["Katarina"], 
				"team": 1
			},
			{
				"champion": champion_name_map["Jinx"], 
				"team": 1
			},
			{
				"champion": champion_name_map["Janna"], 
				"team": 1
			},
		],
		"outcome": False
	}
	example = conv_document_to_examples(exampleGame)
	example_x = example[:-1]
	example_y = example[-1]
	example_xs = np.array([example_x, dup_swap_x(example_x)])
	example_ys = np.array([example_y, float(not example_y)])
	example_yhats = model.predict(example_xs)
	print("Predicted chance blue wins: {}%".format(example_yhats[1]))
	print("Predicted chance red wins: {}%".format(example_yhats[0]))
	print("Averaged predicted chance blue wins: {}%".format((example_yhats[1] + (1-example_yhats[0]))/2))