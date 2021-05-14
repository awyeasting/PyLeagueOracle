import requests
import time
import os
import random

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
def conv_document_to_comp_example(document):
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

# Convert document to a one hot encoding of champions on each team
def conv_document_to_dual_example(document):
	OHEs = np.zeros((10,n_champs))
	i = 0
	for player in document['players']:
		champion = internal_champion_map[player['champion']]
		OHEs[i][champion] = 1
		i += 1
	example = np.append(OHEs, int(document['outcome']))
	return example

def conv_document_to_stats_examples(document):
	examples = []
	for player in document['players']:
		champion = internal_champion_map[player['champion']]
		champ_OHE = np.zeros(n_champs)
		champ_OHE[champion] = 1
		ys = np.array([player['stats'][key] for key in training_config.STATS_KEYS])
		examples.append(np.concatenate((champ_OHE, ys)))
	return examples

def conv_dual_to_comp_example(dual_example):
	exampleX1 = np.zeros(n_champs)
	exampleX2 = np.zeros(n_champs)
	for i in range(5):
		exampleX1 += dual_example[i*n_champs:(i+1)*n_champs]
	for i in range(5,10):
		exampleX2 += dual_example[i*n_champs:(i+1)*n_champs]
	exampleX = np.concatenate((exampleX1, exampleX2))
	return np.append(exampleX, dual_example[-1])

def dual_to_comp_examples(dual_examples):
	return np.array([conv_dual_to_comp_example(dual_example) for dual_example in dual_examples])

def get_comp_examples(documents):
	return [conv_document_to_comp_example(document) for document in documents]

def get_stats_examples(documents):
	examples = []
	for document in documents:
		dexamples = conv_document_to_stats_examples(document)
		for dexample in dexamples:
			examples.append(dexample)
	examples = np.array(examples)
	X = examples[:,:-len(training_config.STATS_KEYS)]
	y = examples[:,-len(training_config.STATS_KEYS):]
	# Normalize ys
	y = y / y.max(axis=0)

	return X, y

# Boost the training data set size by making the model team side invariant
def boost_comp_training_data(X_train, y_train):
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

def get_stats_data(examples):
	X = examples[:,:-len(training_config.STATS_KEYS)]
	y = examples[:,-len(training_config.STATS_KEYS):]

	print("Normalizing stats examples ys")
	# Normalize ys
	y = y / y.max(axis=0)

	print("Splitting stats model examples into train and test...", flush=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		test_size=training_config.STATS_TEST_PORTION,
		random_state=training_config.STATS_SPLIT_SEED)

	return X_train, X_test, y_train, y_test

def get_comp_data(comp_examples):
	X = comp_examples[:,:-1]
	X = X.reshape(-1,10,n_champs)
	y = comp_examples[:,-1]

	print("Splitting dual model examples into train and test...", flush=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
		test_size=training_config.COMP_TEST_PORTION, 
		random_state=training_config.COMP_SPLIT_SEED)

	return X_train, X_test, y_train, y_test

# Get n matches (all matches in database if n < 0)
def get_matches(n = -1):
	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	matchCol = loldb[MATCH_COL_NAME]

	cursor = matchCol.find({"seed_rank.rankMapping":{"$lte":training_config.MAX_RANK_MAPPING_CONSIDERED}})
	matches = []
	for document in cursor:
		matches.append(document)
		if len(matches) % 10000 == 0:
			print("{:,} matches downloaded from database...\n".format(len(matches)), flush=True)
		if n >= 0:
			if len(matches) >= n:
				return matches
	print("{:,} matches downloaded from database".format(len(matches)), flush=True)
	return matches

def get_training_examples(n=-1):
	dual_examples = []
	stats_examples = []

	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	matchCol = loldb[MATCH_COL_NAME]

	cursor = matchCol.find({"seed_rank.rankMapping":{"$lte":training_config.MAX_RANK_MAPPING_CONSIDERED}})
	i = 0
	random.seed(training_config.PORTION_SEED)
	for document in cursor:
		# Monte Carlo approximation of proper splitting works fine for handling a large unknown number of documents
		if random.random() < training_config.COMP_PORTION:
			dual_examples.append(conv_document_to_dual_example(document))
		else:
			examples = conv_document_to_stats_examples(document)
			stats_examples += examples
		i+=1
		if i % 10000 == 0:
			print("{:,} matches downloaded from database...\n".format(i), flush=True)
		if n >= 0:
			if i >= n:
				return np.array(dual_examples), np.array(stats_examples)
	print("{:,} matches downloaded from database".format(i), flush=True)
	return np.array(dual_examples), np.array(stats_examples)

# Swap the teams data in an example
def dup_swap_x(x):
	return np.concatenate((x[n_champs:],x[:n_champs]))

# Swap the teams data in a dual example
def dup_swap_dualx(x):
	return np.concatenate((x[5:],x[:5]))

# Plot the history of a metric for training and testing
def plot_metric_history(model, history, metric_name, num_epochs):
	metric = history.history[metric_name]
	val_metric = history.history['val_' + metric_name]

	e = range(1, num_epochs + 1)

	plt.figure()
	plt.plot(e, metric, 'b', label='Train ' + metric_name)
	plt.plot(e, val_metric, 'r', label='Testing ' + metric_name)
	plt.xlabel('Epoch number')
	plt.ylabel(metric_name)
	plt.title('Comparing training and testing ' + metric_name + ' for ' + model.name)
	plt.legend()
	plt.savefig(model.name + metric_name + '.png')

# Find the best epoch in the model for test set loss
def best_epoch(history, has_acc=False):
	me = np.argmin(history.history['val_loss']) + 1
	if has_acc:
		print("Minimum test loss reached in epoch {} with validation accuracy {}".format(me, history.history['val_accuracy'][me-1]))
	else:
		print("Minimum test loss reached in epoch {}".format(me))
	return me

# Create a sequential model to train on
def create_simple_comp_training_model():
	# Combine one hot encodings within each team to form model input
	inputs = keras.Input(shape=(10,n_champs))
	champTensors = tf.unstack(inputs, 10, axis=1)
	t1 = tf.stack(champTensors[:5], axis=1)
	t2 = tf.stack(champTensors[5:], axis=1)
	t1CompInput = tf.math.reduce_sum(t1, axis=1)
	t2CompInput = tf.math.reduce_sum(t2, axis=1)
	compInput = tf.concat([t1CompInput, t2CompInput], 1)

	# Build fully connected feedforward model
	hnos = [None] * len(training_config.COMP_HIDDEN_NODES)
	hnos[0] = keras.layers.Dense(training_config.COMP_HIDDEN_NODES[0], activation='relu', kernel_initializer='he_normal')(compInput)
	for i in range(1,len(hnos)):
		hnos[i] = keras.layers.Dense(training_config.DUAL_HIDDEN_NODES[i], activation='relu', kernel_initializer='he_normal')(hnos[i-1])
	outputs = keras.layers.Dense(1, activation='sigmoid')(hnos[-1])

	model = keras.Model(inputs=inputs, outputs=outputs, name="CompModel")

	opt = keras.optimizers.Adam(learning_rate=training_config.COMP_LEARNING_RATE)
	model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

	return model

def create_stats_model():
	model = keras.Sequential(name="StatsModel")
	model.add(keras.layers.Dense(training_config.STATS_HIDDEN_NODES[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_champs,)))
	for hn in training_config.STATS_HIDDEN_NODES[1:]:
		model.add(keras.layers.Dense(hn, activation='relu', kernel_initializer='he_normal'))
	model.add(keras.layers.Dense(len(training_config.STATS_KEYS)))

	opt = keras.optimizers.Adam(learning_rate=training_config.STATS_LEARNING_RATE)
	model.compile(optimizer=opt, loss='mae')

	return model

def create_dual_model(stats_model):
	inputs = keras.Input(shape=(10,n_champs))

	# Stat prediction
	champTensors = tf.unstack(inputs, 10, axis=1)
	champStatsTensors = [stats_model(champTensor) for champTensor in champTensors]
	t1ChampStatsTensors = champStatsTensors[:5]
	t2ChampStatsTensors = champStatsTensors[5:]

	t1ChampStatsAvg = tf.math.reduce_mean(t1ChampStatsTensors, axis=0)
	t1ChampStatsMax = tf.math.reduce_max(t1ChampStatsTensors, axis=0)
	t1ChampStatsMin = tf.math.reduce_min(t1ChampStatsTensors, axis=0)

	t2ChampStatsAvg = tf.math.reduce_mean(t2ChampStatsTensors, axis=0)
	t2ChampStatsMax = tf.math.reduce_max(t2ChampStatsTensors, axis=0)
	t2ChampStatsMin = tf.math.reduce_min(t2ChampStatsTensors, axis=0)

	t1ChampStats = tf.concat([t1ChampStatsAvg,t1ChampStatsMin,t1ChampStatsMax], 1)
	t2ChampStats = tf.concat([t2ChampStatsAvg,t2ChampStatsMin,t2ChampStatsMax], 1)

	#print("Input shape:",inputs.shape, flush=True)
	#print("Champ Tensors len:", len(champTensors))
	#print("Champ Tensors shape:",champTensors[0].shape, flush=True)
	#print("Champ stats tensors shape:",champStatsTensors[0].shape, flush=True)
	#print("T1 champ stats len:", len(champStatsTensors))
	#print("T1 champ stats shape:",t1ChampStatsTensors[0].shape, flush=True)
	#print("T1 avg stats shape:",t1ChampStatsAvg.shape, flush=True)
	#print("T1 max stats shape:",t1ChampStatsMax.shape, flush=True)
	#print("T1 min stats shape:",t1ChampStatsMin.shape, flush=True)
	#print("T1 stats shape:",t1ChampStats.shape, flush=True)

	t1 = tf.stack(champTensors[:5], axis=1)
	t2 = tf.stack(champTensors[5:], axis=1)
	#print("t1 shape:", t1.shape)

	# Normal composition based prediction
	t1CompInput = tf.math.reduce_sum(t1, axis=1)
	t2CompInput = tf.math.reduce_sum(t2, axis=1)
	#print("T1 comp input shape:",t1CompInput.shape, flush=True)
	compInput = tf.concat([t1CompInput, t2CompInput], 1)
	#print("Comp input shape:", compInput.shape, flush=True)
	inputPlusTensor = tf.concat([compInput,t1ChampStats,t2ChampStats], 1)
	#print("Input plus shape:",inputPlusTensor.shape, flush=True)
	hnos = [None] * len(training_config.DUAL_HIDDEN_NODES)
	hnos[0] = keras.layers.Dense(training_config.DUAL_HIDDEN_NODES[0], activation='relu', kernel_initializer='he_normal')(inputPlusTensor)
	#print("Hidden nodes 0 shape:", hnos[0].shape)
	for i in range(1,len(hnos)):
		hnos[i] = keras.layers.Dense(training_config.DUAL_HIDDEN_NODES[i], activation='relu', kernel_initializer='he_normal')(hnos[i-1])
		#print("Hidden nodes {} shape:".format(i), hnos[i].shape)
	outputs = keras.layers.Dense(1, activation='sigmoid')(hnos[-1])
	#print("Outputs shape:", outputs.shape, flush=True)
	#print("")
	model = keras.Model(inputs=inputs, outputs=outputs, name="DualModel")

	opt = keras.optimizers.Adam(learning_rate=training_config.DUAL_LEARNING_RATE)
	model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])

	return model

# Train the model on training data using test data as validation
def train_model(model, X_train, X_test, y_train, y_test, batch_size, num_epochs, save_path, has_acc = False):
	if batch_size < 0:
		batch_size = len(y_train)

	history = model.fit(
		X_train, 
		y_train, 
		epochs = num_epochs, 
		batch_size = batch_size, 
		validation_data=(X_test,y_test),
		verbose = 1)

	model.save_weights(save_path)

	if has_acc:
		loss, acc = model.evaluate(X_test, y_test, verbose = 1)
		return loss, acc, history
	else:
		loss = model.evaluate(X_test, y_test, verbose = 1)
		return loss, None, history

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

# Returns the model's accuracy with least confident predictions using multiple confidence levels
def get_unconfidence_accuracies(model, X, y, confidenceLevels=[]):
	if not len(confidenceLevels):
		print("Error: Please provide a confidence level to get the accuracy of")
		return None

	# Sort model predictions by most confident
	yhats = model.predict(X)
	yhats_confidences = []
	for i, yhat in enumerate(yhats):
		yhats_confidences.append(((yhat-0.5)**2, yhat, y[i]))
	yhats_confidences.sort(key=lambda x:x[0],reverse=False)

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

def predict_game(model, document, display=False):
	# Add outcome to game if not there just to make sure conv_document_to_examples doesn't freak out
	if not document.get("outcome"):
		document["outcome"] = False

	example = conv_document_to_dual_example(document)
	example_x = example[:-1]
	example_x = example_x.reshape(10, n_champs)
	example_xs = np.array([example_x, dup_swap_dualx(example_x)])
	example_yhats = model.predict(example_xs)

	blueChance = example_yhats[1][0]
	redChance = example_yhats[0][0]
	avgBlueChance = (blueChance + (1-redChance))/2

	if display:
		print("{:.2f}%\tPredicted chance blue wins".format(100*blueChance))
		print("{:.2f}%\tPredicted chance red wins".format(100*redChance))
		print("--------------------------------------------")
		print("{:.2f}%\tAveraged predicted chance blue wins".format(100*avgBlueChance))
		print("{:.2f}%\tAveraged predicted chance red wins".format(100*(1 - avgBlueChance)), flush=True)

	return blueChance, redChance, avgBlueChance