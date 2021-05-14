COMP_HIDDEN_NODES = [100]
COMP_LEARNING_RATE = 0.00005
COMP_NUM_EPOCHS = 700
COMP_BATCH_SIZE = -1
COMP_TEST_PORTION = 0.2
COMP_SPLIT_SEED = 1338

COMP_PORTION = 0.8 # How much of the training data to give to the composition model vs stats model
PORTION_SEED = 1337

STATS_HIDDEN_NODES = [128]
STATS_LEARNING_RATE = 0.0005
STATS_NUM_EPOCHS = 600
STATS_BATCH_SIZE = -1
STATS_TEST_PORTION = 0.2
STATS_SPLIT_SEED = 1338

DUAL_HIDDEN_NODES = [100]
DUAL_LEARNING_RATE = 0.00005
DUAL_NUM_EPOCHS = 500
DUAL_BATCH_SIZE = -1

N_MATCHES = -1

DO_MATCH_DUP = True

DO_COMP_TRAIN = False
DO_COMP_LOAD = True

DO_DUAL_TRAIN = False
DO_DUAL_LOAD = True

DO_STATS_TRAIN = False
DO_STATS_LOAD = True or DO_DUAL_LOAD or DO_DUAL_TRAIN

DO_LOAD_DATA = True or DO_COMP_TRAIN or DO_STATS_TRAIN or DO_DUAL_TRAIN

CHECKPOINT_PATH = "training_1/cp.ckpt"
COMP_SAVE_PATH = "./checkpoints/comp_most_recent"
STATS_SAVE_PATH = "./checkpoints/stats_most_recent"
DUAL_SAVE_PATH = "./checkpoints/dual_most_recent"

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