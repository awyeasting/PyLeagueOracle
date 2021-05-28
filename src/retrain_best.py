import time

from trainer import Trainer
import training_config

if __name__ == '__main__':
	#start_time = time.time()
	print("Retraining best model...", flush=True)
	trainer = Trainer()

	# Load best model
	trainer.load_best_model()

	# Load data
	trainer.load_data()

	# Train data
	trainer.train_load_models()

	# Print stats (for debugging purposes)
	trainer.print_model_info()

	#print("Sleeping for {} hours before retraining again...\n".format(training_config.BEST_RETRAIN_WAIT/(60*60)), flush=True)
	#cur_time = time.time()
	# Remove training time from the wait time so that it retrains at approximately the same time every day
	#wait = start_time + training_config.BEST_RETRAIN_WAIT - cur_time
	#time.sleep(wait)