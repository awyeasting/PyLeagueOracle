import comp_model
import stats_model
import dual_model

# Configurations for data load
COMP_PORTION = 0.8 # How much of the training data to give to the composition model vs stats model
COMP_TEST_PORTION = 0.2
STATS_TEST_PORTION = 0.2
COMP_SPLIT_SEED = 1338
STATS_SPLIT_SEED = 1338
PORTION_SEED = 1337
N_MATCHES = -1

# Configurations for model structure and training properties
MODELS = [
	{
		"NAME" : "CompModel",
		"DO_TRAIN": True,
		"CLASS": comp_model.CompModel,
		"CONFIG": {
			"IS_PREDICT_MODEL": True,
			"HIDDEN_NODES" : [100],
			"LEARNING_RATE" : 0.0005,
			"NUM_EPOCHS" : 1000,
			"BATCH_SIZE" : -1,
			"REG_PARAM" : 0.01,
			"PATIENCE" : 25
		}
	},
	{
		"NAME" : "StatsModel",
		"DO_TRAIN": True,
		"CLASS": stats_model.StatsModel,
		"CONFIG": {
			"IS_PREDICT_MODEL": False,
			"HIDDEN_NODES" : [128],
			"LEARNING_RATE" : 0.0005,
			"NUM_EPOCHS" : 600,
			"BATCH_SIZE" : -1,
			"PATIENCE" : 25
		}
	},
	{
		"NAME" : "DualModel",
		"DO_TRAIN": True,
		"CLASS": dual_model.DualModel,
		"CONFIG": {
			"IS_PREDICT_MODEL": True,
			"HIDDEN_NODES" : [100],
			"LEARNING_RATE" : 0.00005,
			"NUM_EPOCHS" : 1000,
			"BATCH_SIZE" : -1,
			"PATIENCE" : 25,
			"STATS_MODEL_NAME" : "StatsModel"
		}
	}
]


# MISC
DO_MATCH_DUP = False

DO_COMP_TRAIN = True
DO_COMP_LOAD = False

DO_DUAL_TRAIN = False
DO_DUAL_LOAD = False

DO_STATS_TRAIN = False
DO_STATS_LOAD = False or DO_DUAL_LOAD or DO_DUAL_TRAIN

DO_LOAD_DATA = True

CHECKPOINT_PATH = "training_1/cp.ckpt"
BASE_SAVE_PATH = "./checkpoints/{}_most_recent"

BEST_SAVE_PATH = "models/best_model"

BEST_RETRAIN_WAIT = 24 * 60 * 60 # Currently retrains once a day

DO_PREDICT_MANUAL_EXAMPLES = True

# Rank mappings:
# 0: Challenger
# 1: Grandmaster
# 2: Master
# 3: Diamond
# 4: Platinum
# 5: Gold
# 6: Silver
# 7: Bronze
# 8: Iron
MAX_RANK_MAPPING_CONSIDERED = 4

STATS_KEYS = [
	"kills",
	"deaths",
	"assists",
	"totalDamageDealt",
	"magicDamageDealt",
	"physicalDamageDealt",
	"trueDamageDealt",
	"totalDamageDealtToChampions",
	"magicDamageDealtToChampions",
	"physicalDamageDealtToChampions",
	"trueDamageDealtToChampions",
	"totalHeal",
	"totalUnitsHealed",
	"damageSelfMitigated",
	"damageDealtToObjectives",
	"damageDealtToTurrets",
	"timeCCingOthers",
	"totalDamageTaken",
	"magicalDamageTaken",
	"physicalDamageTaken",
	"trueDamageTaken",
	"totalMinionsKilled",
]
