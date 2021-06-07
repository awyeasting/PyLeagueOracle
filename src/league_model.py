from tensorflow import keras

from training import train_model, best_epoch

class LeagueModel:
	def __init__(self, config):
		self.config = config
		self.trained = False
		self.model = None
		self.predictModel = config["IS_PREDICT_MODEL"]
	
	def train(self, data):
		es = keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=self.config["PATIENCE"])

		self.loss, self.acc, self.history = train_model(self.model, 
			data,
			self.config,
			has_acc=self.predictModel,
			cb_list=[es])

		best_epoch(self.history, has_acc=self.predictModel)

		if self.predictModel:
			print('{} test accuracy: {:.3f}'.format(self.model.name, self.acc))

	def load(self):
		self.model.load_weights(BASE_SAVE_PATH.format(self.model.name))