import numpy as np
import time

from flask import Flask, request, abort
from flask_cors import CORS
from tensorflow import keras

import training_config
from training import dup_swap_dualx, conv_document_to_dual_example
from example_games import exampleGames
from champion_map import n_champs

app = Flask(__name__)
CORS(app)
model = keras.models.load_model(training_config.BEST_SAVE_PATH)

def predict_games(model, documents):
	examples = []
	for document in documents:
		# Add outcome to game if not there just to make sure conv_document_to_examples doesn't freak out
		if not document.get("outcome"):
			document["outcome"] = False

		example = conv_document_to_dual_example(document)
		example_x = example[:-1]
		example_x = example_x.reshape(10, n_champs)
		examples.append(example_x)
		examples.append(dup_swap_dualx(example_x))

	examples = np.array(examples)
	preds = model.predict(examples)

	outcomes = []
	for i in range(0,len(preds),2):
		outcome = {}
		outcome["red_chance"] = str(preds[i][0])
		outcome["blue_chance"] = str(preds[i+1][0])
		outcome["avg_blue_chance"] = str((preds[i+1][0] + (1-preds[i][0]))/2)
		outcomes.append(outcome)
	return outcomes

@app.route("/predict", methods=['POST'])
def predict():
	data = request.json
	if not data or not data.get("games"):
		return "", 400

	t1 = time.time()
	outcomes = predict_games(model, data["games"])
	t2 = time.time()
	return {"predictions": outcomes, "prediction_time": str(t2-t1)}