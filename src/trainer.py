import functools
import sys

from tensorflow import keras

from champion_map import internal_champion_map, n_champs, champion_name_map
import training_config
from training import get_training_examples, dual_to_comp_examples
from training import get_comp_data, get_stats_data, train_model, plot_metric_history
from training import best_epoch, get_confidence_accuracies, get_unconfidence_accuracies
from training import predict_game

class Trainer:

	def __init__(self):
		self.models = []
		self.comp_data = None
		self.stats_data = None

	def create_models(self):
		for model in training_config.MODELS:
			if model["DO_TRAIN"]:
				if model["CONFIG"].get("STATS_MODEL_NAME"):
					self.models.append(model["CLASS"](model["NAME"], model["CONFIG"], n_champs, self.models))
				else:
					if model["CONFIG"]["IS_PREDICT_MODEL"]:
						self.models.append(model["CLASS"](model["NAME"], model["CONFIG"], n_champs))
					else:
						self.models.append(model["CLASS"](model["NAME"], model["CONFIG"], n_champs, training_config.STATS_KEYS))

	def load_best_model(self):
		try:
			best_model = keras.models.load_model(training_config.BEST_SAVE_PATH)
			self.models.append(best_model)
		except:
			print("Failed to load best model (no best model saved yet?)")

	def load_data(self):
		# If there are any untrained models loaded then get the training data
		for model in self.models:
			if not model.trained:
				self.comp_examples, self.stats_examples = get_training_examples(training_config.N_MATCHES)

				self.comp_data = get_comp_data(self.comp_examples)
				self.stats_data = get_stats_data(self.stats_examples)
				return

	def train_models(self):
		for model in self.models:
			if not model.trained:
				if model.predictModel:
					model.train(self.comp_data)
				else:
					model.train(self.stats_data)

	def print_model_info(self):
		# Print model summaries
		for model in self.models:
			model.model.summary()

			if model.predictModel:
				print("\nConfidence accuracies for {}".format(model.model.name))
				# Print most confident accuracies
				confidenceLevels = [1, 1/10, 1/100]
				conf_accs, max_conf = get_confidence_accuracies(model.model, self.comp_data, confidenceLevels=confidenceLevels)
				for i in range(len(confidenceLevels)):
					if confidenceLevels[i] == 1:
						print("Average predictions accuracy:\t\t\t{:.2f}%".format(100*conf_accs[i]) )
					else:
						print("{:.2f}% most confident predictions accuracy:\t{:.2f}%".format(confidenceLevels[i]* 100, 100*conf_accs[i]) )
					print("Max confidence at this level:\t\t\t\t{:.2f}%".format(100*max_conf[i]))
				# Print least confident accuracies
				uconfidenceLevels = [1/10, 1/100]
				uconf_accs = get_unconfidence_accuracies(model.model, self.comp_data, confidenceLevels=uconfidenceLevels)
				for i in range(len(uconfidenceLevels)):
					print("{:.2f}%% least confident predictions accuracy:\t{:.2f}%".format(uconfidenceLevels[i]* 100, 100*uconf_accs[i]) )

	def save_best_model(self):
		if training_config.DO_LOAD_DATA:
			print("\nLoading past best model (if exists)...")
			try:
				best_model = keras.models.load_model(training_config.BEST_SAVE_PATH)
				best_loss, _ = best_model.evaluate(self.comp_data["X_test"], self.comp_data["y_test"], batch_size=len(self.comp_data["y_test"]))
			except:
				best_model = None
				best_loss = sys.float_info.max

			for model in self.models:
				if model.predictModel:
					print("Testing new model...")
					loss, acc = model.model.evaluate(self.comp_data["X_test"], self.comp_data["y_test"], batch_size=len(self.comp_data["y_test"]))
					if loss < best_loss:
						best_model = model.model
						best_loss = loss
						print("New best model found!")

			best_model.save(training_config.BEST_SAVE_PATH)
			print("\nBest model loss: ", best_loss)

if __name__ == '__main__':
	trainer = Trainer()

	trainer.create_models()

	trainer.load_data()

	trainer.train_models()

	trainer.print_model_info()

	trainer.save_best_model()