import requests
#import csv
import numpy as np

import pulldata
import time

# TODO: Return to this with own data and make a system for manually labeling top, jungle, mid, bot, support to obtain training data
POSITION_MAPPING_TRAINING_DATA = "http://raw.githubusercontent.com/Canisback/roleML/master/data/verification_results.csv"
# POSITION_MAPPING_TRAINING_FILE_NAME = "verification_results.csv"

# One hot encoding of position label
BASIC_POSITION_MAP = {
	"toplaner": [1,0,0,0,0],
	"jungler":  [0,1,0,0,0],
	"midlaner": [0,0,1,0,0],
	"carry":    [0,0,0,1,0],
	"support":  [0,0,0,0,1]
}
BASIC_POSITION_ENUM = {
	"toplaner": 1,
	"jungler":  2,
	"midlaner": 3,
	"carry":    4,
	"support":  5
}


# Pull training data from url
# Tie matches use riot api to get 
def transform_position_mapping_data():
	response = requests.get(POSITION_MAPPING_TRAINING_DATA)
	training = [line.strip().split(',') for line in response.text.splitlines()]
	header = training[0]
	training = training[1:]

	# Remove game Id and the row index
	header = header[1:-1]
	# Convert player number labels to integers and ascending
	header = [int(pid) for pid in header]
	resort = np.argsort(header)

	training = np.array(training)
	# Take match_ids and row numbers off training data and sort players in ascending player number order
	match_ids = training[:, -1]
	training = training[:, 1:-1]
	training = training[:, resort]
	training = training.tolist()

	assert len(match_ids) == len(training)

	good_data, bad_indices = [], []
	for i, line in enumerate(training):
		try:
			good_data.append(
				[BASIC_POSITION_ENUM[p] for p in line]
			)
		except KeyError:
			bad_indices.append(i)
	training = np.array(good_data)
	match_ids = np.array([match_id for i, match_id in enumerate(match_ids) if i not in bad_indices])

	#print(match_ids)

	match_ids = match_ids.astype(float)
	#print(match_ids)
	match_ids = match_ids.astype('int64')
	assert len(match_ids) == len(training)

	#print(match_ids)
	#print(training)

	# TODO: Get matches from match id and add champion data to internal data and return
	print("Have position training data")
	print("Getting match data...")
	i = 1
	matches = []
	for match_id in match_ids:
		print ("Pulling match {}/{}".format(i,len(match_ids)))
		matches.append(pulldata.pull_match_data(match_id))
		i+=1
		# 120 seconds / 100 requests
		time.sleep(120/100)
	print("All matches pulled")

	X = []
	y = []
	return X, y

# TODO train support vector machine or other such algorithm to classify champion positions
# Single X data: one hot encoded representation of the champion at question followed by a combined one hot encoding of the rest of the team's champions (Size: 310)
# Single Y data: one hot encoded representation of the champion's position on the team (Size: 5)
def train_clf(X, y):
	pass

if __name__ == "__main__":
	X, y = transform_position_mapping_data()
	clf = train_clf(X,y)
