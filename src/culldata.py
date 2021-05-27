import time

import pymongo

from HIDDEN_CONFIG import MONGODB_CONN_STR
from config import LOL_DB_NAME, MATCH_COL_NAME, MAX_MATCH_AGE_DAYS, MATCH_CULL_WAIT

def cull_many_matches(match_col):
	oldestTimeStamp = 1000*(time.time() - (MAX_MATCH_AGE_DAYS * 24 * 60 * 60))
	result = match_col.delete_many({"gameCreation":{"$lt":int(oldestTimeStamp)}})
	print("\t{} old matches culled".format(result.deleted_count))

if __name__ == "__main__":
	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	match_col = loldb[MATCH_COL_NAME]

	while True:
		print("Culling old matches...", flush=True)
		cull_many_matches(match_col)
		print("Sleeping for {} hours before culling again...\n".format(MATCH_CULL_WAIT/(60*60)), flush=True)
		time.sleep(MATCH_CULL_WAIT)
