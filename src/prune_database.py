import pymongo

from HIDDEN_CONFIG import MONGODB_CONN_STR

from config import LOL_DB_NAME, MATCH_COL_NAME

if __name__ == '__main__':
	dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
	loldb = dbclient[LOL_DB_NAME]
	match_col = loldb[MATCH_COL_NAME]

	# For an older version of the code which stored redundant player information in teams and players
	match_col.update_many({},{'$unset':{'teams':1}})