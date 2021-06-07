import tensorflow as tf
from tensorflow import keras

import league_model
from training import train_model, best_epoch

class StatsModel(league_model.LeagueModel):
	def __init__(self, name, config, n_champs, stats_keys):
		super().__init__(config)

		self.model = keras.Sequential(name=name)
		self.model.add(keras.layers.Dense(config["HIDDEN_NODES"][0], activation='relu', kernel_initializer='he_normal', input_shape=(n_champs,)))
		for hn in config["HIDDEN_NODES"][1:]:
			self.model.add(keras.layers.Dense(hn, activation='relu', kernel_initializer='he_normal'))
		self.model.add(keras.layers.Dense(len(stats_keys)))

		opt = keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
		self.model.compile(optimizer=opt, loss='mae')