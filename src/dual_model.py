import tensorflow as tf
from tensorflow import keras

import league_model
from stats_model import StatsModel
from training import train_model

class DualModel(league_model.LeagueModel):
	def __init__(self, name, config, n_champs, models):
		super().__init__(config)

		inputs = keras.Input(shape=(10,n_champs))

		for model in models:
			if model.model.name == config["STATS_MODEL_NAME"]:
				stats_model = model.model
				break

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

		t1 = tf.stack(champTensors[:5], axis=1)
		t2 = tf.stack(champTensors[5:], axis=1)

		# Normal composition based prediction
		t1CompInput = tf.math.reduce_sum(t1, axis=1)
		t2CompInput = tf.math.reduce_sum(t2, axis=1)
		compInput = tf.concat([t1CompInput, t2CompInput], 1)
		inputPlusTensor = tf.concat([compInput,t1ChampStats,t2ChampStats], 1)
		hnos = [None] * len(config["HIDDEN_NODES"])
		hnos[0] = keras.layers.Dense(config["HIDDEN_NODES"][0], activation='relu', kernel_initializer='he_normal')(inputPlusTensor)
		for i in range(1,len(hnos)):
			hnos[i] = keras.layers.Dense(config["HIDDEN_NODES"][i], activation='relu', kernel_initializer='he_normal')(hnos[i-1])
		outputs = keras.layers.Dense(1, activation='sigmoid')(hnos[-1])
		
		self.model = keras.Model(inputs=inputs, outputs=outputs, name=name)

		opt = keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
		self.model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])