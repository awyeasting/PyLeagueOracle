import time

import pymongo

import champion_map

from HIDDEN_CONFIG import API_KEY, MONGODB_CONN_STR
from config import LOL_DB_NAME, MATCH_COL_NAME, SEED_COL_NAME, USE_SEEDS_COL
from matches import pull_many_matches

# Print the match information in human readable format
# (for debugging purposes)
def print_match(players, outcome):
	for side in [0,1]:
		if side:
			print("Red Team:")
		else:
			print("Blue Team:")
		for player in players:
			if side == player["team"]:
				print("\t",champion_map.champion_map[player["champion"]], player["role"], player["lane"])
	if outcome:
		print("Red side win")
	else:
		print("Blue side win")
	print("",flush=True)

if __name__ == "__main__":
	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	match_col = loldb[MATCH_COL_NAME]
	seed_col = None
	if USE_SEEDS_COL:
		seed_col = loldb[SEED_COL_NAME]

	while True:
		print("Pulling matches...\n",flush=True)
		try:
			pull_many_matches(matchCol=match_col, seedCol=seed_col)
		except:
			print("Pull many matches failed. Waiting 10 seconds before retry...")
			# Wait 10 seconds before restarting match pulling if there was an unexpected error
			time.sleep(10)
