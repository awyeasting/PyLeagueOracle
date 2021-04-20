import requests
import csv

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

# Pull training data from url
# Tie matches use riot api to get 
def transform_position_mapping_data():
	response = requests.get(POSITION_MAPPING_TRAINING_DATA)
	for line in response.text.splitlines():
		# TODO: Read line into internal data structure
		print(line)
	# TODO: Get matches from match id and add champion data to internal data and return
	X = []
	y = []
	return X, y

# TODO train support vector machine to classify champion positions
def train_clf(X, y):
	pass

if __name__ == "__main__":
	X, y = transform_position_mapping_data()
	clf = train_clf(X,y)
