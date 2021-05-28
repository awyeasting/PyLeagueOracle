import functools
import sys

from tensorflow import keras

from champion_map import internal_champion_map, n_champs, champion_name_map

import training_config
from training import create_simple_comp_training_model, create_stats_model, create_dual_model
from training import get_training_examples, dual_to_comp_examples
from training import get_comp_data, get_stats_data, train_model, plot_metric_history
from training import best_epoch, get_confidence_accuracies, get_unconfidence_accuracies
from training import predict_game

class Trainer:
	def __init__(self):
		self.comp_model = training_config.DO_COMP_TRAIN or training_config.DO_COMP_LOAD
		self.stats_model = training_config.DO_STATS_TRAIN or training_config.DO_STATS_LOAD
		self.dual_model = training_config.DO_DUAL_TRAIN or training_config.DO_DUAL_LOAD
		self.best_model = None

		self.stats_examples = None
		self.comp_examples = None
		self.predictModels = []
		self.isPredictModel = False

	def create_models(self):
		# Create models desired
		if self.comp_model:
			self.comp_model = create_simple_comp_training_model()
		if self.stats_model:
			self.stats_model = create_stats_model()

			# Dual model dependent on there being a stats model
			if self.dual_model:
				self.dual_model = create_dual_model(self.stats_model)

		# Models used for making a prediction on a match using only compositional data
		self.predictModels.append(self.comp_model)
		self.predictModels.append(self.dual_model)

		# Models used for making a prediction 
		self.isPredictModel = functools.reduce(lambda a,b: a or b, map(lambda x: bool(x), self.predictModels))

	def load_best_model(self):
		try:
			self.best_model = keras.models.load_model(training_config.BEST_SAVE_PATH)
		except:
			self.best_model = None

		self.predictModels.append(self.best_model)
		self.isPredictModel = self.isPredictModel or bool(self.best_model)

	def load_data(self):
		if training_config.DO_LOAD_DATA:
			self.comp_examples, self.stats_examples = get_training_examples(training_config.N_MATCHES)

			if self.isPredictModel:
				self.compX_train, self.compX_test, self.compy_train, self.compy_test = get_comp_data(self.comp_examples)

	def train_load_models(self):
		# Remove models not created already
		if type(self.comp_model) == bool:
			self.comp_model = None
		if type(self.dual_model) == bool:
			self.dual_model = None
		if type(self.stats_model) == bool:
			self.stats_model = None

		# Train or load the models
		if self.comp_model:
			if training_config.DO_COMP_TRAIN:
				self.comp_loss, self.comp_acc, self.comp_history = train_model(self.comp_model, 
					self.compX_train, 
					self.compX_test, 
					self.compy_train, 
					self.compy_test, 
					training_config.COMP_BATCH_SIZE, 
					training_config.COMP_NUM_EPOCHS,
					training_config.COMP_SAVE_PATH,
					has_acc=True)

				plot_metric_history(self.comp_model, self.comp_history, 'loss', training_config.COMP_NUM_EPOCHS)
				plot_metric_history(self.comp_model, self.comp_history, 'accuracy', training_config.COMP_NUM_EPOCHS)
				
				best_epoch(self.comp_history, has_acc=True)

				print('Comp test accuracy: %.3f' % self.comp_acc)
			elif training_config.DO_COMP_LOAD:
				self.comp_model.load_weights(training_config.COMP_SAVE_PATH)

		if self.stats_model:
			if training_config.DO_STATS_TRAIN:
				
				self.statsX_train, self.statsX_test, self.statsy_train, self.statsy_test = get_stats_data(self.stats_examples)

				self.stats_loss, self.stats_acc, self.stats_history = train_model(self.stats_model, 
					self.statsX_train, 
					self.statsX_test, 
					self.statsy_train, 
					self.statsy_test,
					training_config.STATS_BATCH_SIZE, 
					training_config.STATS_NUM_EPOCHS,
					training_config.STATS_SAVE_PATH)

				plot_metric_history(self.stats_model, self.stats_history, 'loss', training_config.STATS_NUM_EPOCHS)

				best_epoch(self.stats_history)

			elif training_config.DO_STATS_LOAD:
				self.stats_model.load_weights(training_config.STATS_SAVE_PATH)

		if self.dual_model:
			if training_config.DO_DUAL_TRAIN:

				self.dual_loss, self.dual_acc, self.dual_history = train_model(self.dual_model, 
					self.compX_train, 
					self.compX_test, 
					self.compy_train, 
					self.compy_test, 
					training_config.DUAL_BATCH_SIZE, 
					training_config.DUAL_NUM_EPOCHS,
					training_config.DUAL_SAVE_PATH,
					has_acc=True)

				plot_metric_history(self.dual_model, self.dual_history, 'loss', training_config.DUAL_NUM_EPOCHS)
				plot_metric_history(self.dual_model, self.dual_history, 'accuracy', training_config.DUAL_NUM_EPOCHS)
				
				best_epoch(self.dual_history, has_acc=True)

				print('Dual test accuracy: %.3f' % self.dual_acc)
			elif training_config.DO_DUAL_LOAD:
				self.dual_model.load_weights(training_config.DUAL_SAVE_PATH)

		if training_config.DO_LOAD_DATA and self.best_model:
			es = keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=50)

			best_loss, best_acc, best_history = train_model(self.best_model, 
				self.compX_train, 
				self.compX_test, 
				self.compy_train, 
				self.compy_test, 
				training_config.BEST_BATCH_SIZE, 
				training_config.BEST_NUM_EPOCHS,
				training_config.BEST_SAVE_PATH,
				has_acc=True,
				cb_list=[es])

			print('Best test loss: {}%\nBest test accuracy: {}%'.format(best_loss, best_acc))


	def print_model_info(self):
		# Print model summaries
		if self.stats_model:
			self.stats_model.summary()
		for model in self.predictModels:
			if model:
				model.summary()

		# Print model effectiveness if there's data to test that on
		if training_config.DO_LOAD_DATA:
			for model in self.predictModels:
				if model:
					print("\nConfidence accuracies for {}".format(model.name))
					# Print most confident accuracies
					confidenceLevels = [1, 1/10, 1/100]
					conf_accs, max_conf = get_confidence_accuracies(model, self.compX_test, self.compy_test, confidenceLevels=confidenceLevels)
					for i in range(len(confidenceLevels)):
						if confidenceLevels[i] == 1:
							print("Average predictions accuracy:\t\t\t{:.2f}%".format(100*conf_accs[i]) )
						else:
							print("{:.2f}% most confident predictions accuracy:\t{:.2f}%".format(confidenceLevels[i]* 100, 100*conf_accs[i]) )
						print("Max confidence at this level:\t\t\t\t{:.2f}%".format(100*max_conf[i]))
					# Print least confident accuracies
					uconfidenceLevels = [1/10, 1/100]
					uconf_accs = get_unconfidence_accuracies(model, self.compX_test, self.compy_test, confidenceLevels=uconfidenceLevels)
					for i in range(len(uconfidenceLevels)):
						print("{:.2f}%% least confident predictions accuracy:\t{:.2f}%".format(uconfidenceLevels[i]* 100, 100*uconf_accs[i]) )

	def predict_manual_examples(self):
		if self.isPredictModel and training_config.DO_PREDICT_MANUAL_EXAMPLES:
			print("\nTesting example games...")
			from example_games import exampleGames
			for i, exampleGame in enumerate(exampleGames):
				print("Predicting example game {}".format(i))
				for model in self.predictModels:
					if model:
						print("\n{} prediction:".format(model.name))
						blueChance, redChance, avgBlueChance = predict_game(model, exampleGame, display=True)

	def save_best_model(self):
		if training_config.DO_LOAD_DATA:
			print("\nLoading past best model (if exists)...")
			try:
				best_model = keras.models.load_model(training_config.BEST_SAVE_PATH)
				best_loss, _ = model.evaluate(self.compX_test, self.compy_test, len(self.compy_test))
			except:
				best_model = None
				best_loss = sys.float_info.max

			for model in self.predictModels:
				if model:
					print("Testing new model...")
					loss, acc = model.evaluate(self.compX_test, self.compy_test, batch_size=len(self.compy_test))
					if loss < best_loss:
						best_model = model
						best_loss = loss
						print("New best model found!")

			best_model.save(training_config.BEST_SAVE_PATH)
			print("\nBest model loss: ", best_loss)

if __name__ == '__main__':
	trainer = Trainer()

	trainer.create_models()

	trainer.load_data()

	trainer.train_load_models()

	trainer.print_model_info()

	trainer.predict_manual_examples()

	trainer.save_best_model()