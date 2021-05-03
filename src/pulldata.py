import pymongo

import champion_map

from HIDDEN_CONFIG import API_KEY, MONGODB_CONN_STR
from config import LOL_DB_NAME, MATCH_COL_NAME
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

	# Default to pulling matches from challenger instead of starting from a seed account
	pull_many_matches(matchCol=match_col)