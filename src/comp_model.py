import tensorflow as tf
from tensorflow import keras

import league_model
from training import train_model

class CompModel(league_model.LeagueModel):
	def __init__(self, name, config, n_champs):
		super().__init__(config)

		# Combine one hot encodings within each team to form model input
		inputs = keras.Input(shape=(10,n_champs))
		champTensors = tf.unstack(inputs, 10, axis=1)
		t1 = tf.stack(champTensors[:5], axis=1)
		t2 = tf.stack(champTensors[5:], axis=1)
		t1CompInput = tf.math.reduce_sum(t1, axis=1)
		t2CompInput = tf.math.reduce_sum(t2, axis=1)
		compInput = tf.concat([t1CompInput, t2CompInput], 1)

		# Build fully connected feedforward model
		hnos = [None] * len(config["HIDDEN_NODES"])
		hnos[0] = keras.layers.Dense(config["HIDDEN_NODES"][0], 
			activation='relu', 
			kernel_initializer='he_normal', 
			kernel_regularizer=keras.regularizers.l2(l=config["REG_PARAM"]))(compInput)
		for i in range(1,len(hnos)):
			hnos[i] = keras.layers.Dense(config["HIDDEN_NODES"][i], 
				activation='relu', 
				kernel_initializer='he_normal',
				kernel_regularizer=keras.regularizers.l2(l=config["REG_PARAM"]))(hnos[i-1])
		outputs = keras.layers.Dense(1, activation='sigmoid')(hnos[-1])

		self.model = keras.Model(inputs=inputs, outputs=outputs, name=name)

		opt = keras.optimizers.Adam(learning_rate=config["LEARNING_RATE"])
		self.model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])