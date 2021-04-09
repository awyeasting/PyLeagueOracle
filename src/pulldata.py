import requests
import time

import numpy as np
import pymongo

import champion_map

from HIDDEN_CONFIG import API_KEY, MONGODB_CONN_STR

dbclient = pymongo.MongoClient(MONGODB_CONN_STR)
loldb = dbclient["leagueoflegends"]
match_col = loldb["matches"]

# Keys expire in 24 hours
# 20 requests / 1 second
# 100 requests / 2 minutes
RIOT_API_BASE_URL = "https://americas.api.riotgames.com"
LOL_API_BASE_URL = "https://na1.api.riotgames.com"
# match-v4: /lol/match/v4/matches/{matchId}		Get match by match ID
# match-v4: /lol/match/v4/matchlists/by-account/{encryptedAccountId}

# I'm not kidding, this is actually Riot game's constant for the ranked solo/duo queue games
RANKED_QUEUE_ID = 420

# API NOTES:
# Three ids: summoner id, account id, puuid
# summoner and account ids are unique per region
# puuids are unique globally

# process for getting encrypted account id
# /riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine} -> encrypted PUUID
GET_PUUID_PATH = "/riot/account/v1/accounts/by-riot-id/"
# /lol/summoner/v4/summoners/by-puuid/{encryptedPUUID} -> encrypted Account ID
GET_ACCOUNT_PATH = "/lol/summoner/v4/summoners/by-puuid/"
# Nubrozaref#NA1 current encrypted puuid:	"6Uf6wy09tg40GDDz6VQ2WfUi63aDPBuWZMQn4xCzFRo03MvskmHpkHzPlOHfLgmfclV4MItRLoZRvg"
# 				current encrypted account id: "eJwW8oKNquGnNm6CcyxlQpZt6MiRhcApI-i162LOh-rVoA"

# process for getting match data:
# /lol/match/v4/matchlists/by-account/{encryptedAccountId} (queue = 420 for solo queue) -> [gameId]
GET_MATCHES_PATH = "/lol/match/v4/matchlists/by-account/"
# /lol/match/v4/matches/{matchId} ->
GET_MATCH_PATH = "/lol/match/v4/matches/"

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

# Pull
def pull_match_data(matchId):
	match_data = {}
	url = LOL_API_BASE_URL + GET_MATCH_PATH + str(matchId) + "?api_key=" + API_KEY
	response = requests.get(url)
	match = response.json()

	# Get player non specific data
	match_data["gameId"] = match["gameId"]
	match_data["gameCreation"] = match["gameCreation"]
	match_data["gameDuration"] = match["gameDuration"]

	# Get player specific data
	teams = [[],[]]
	players = []
	for pIden in match["participantIdentities"]:
		player = {}
		player["id"] = pIden["participantId"]
		p = match["participants"][player["id"] - 1]
		player["team"] = int(200 == p["teamId"])
		player["champion"] = p["championId"]
		player["role"] = p["timeline"]["role"]
		player["lane"] = p["timeline"]["lane"]
		teams[player["team"]].append(player)
		players.append(player)
	match_data["teams"] = teams

	# Get match outcome
	outcome = match["teams"][0]["win"] == "Fail"
	if match["teams"][0]["teamId"] == 200:
		outcome = not outcome
	match_data["outcome"] = outcome

	# Print match data for debugging purposes
	print_match(players, outcome)
	return match_data

# Pull recent ranked matched associated with a specific encryptedAccountId
def pull_players_matches(encryptedAccountId):
	url = LOL_API_BASE_URL + GET_MATCHES_PATH + encryptedAccountId + "?queue=420&api_key=" + API_KEY
	response = requests.get(url)
	matches = response.json()["matches"]
	print("Found",len(matches),"matches")
	matches_data = []
	for match in matches:
		# TODO: check if match is already in the database before grabbing match data
		print("Getting match data...")
		match_data = pull_match_data(match["gameId"])
		# TODO: extract players from match to use as seed players for more matches
		# TODO: save match in database
		matches_data.append(match_data)
		print("Added match data")
		print(match_data, flush=True)
		time.sleep(100)
	return matches

def pull_encrypted_account_id(summonerName, tagLine):
	url = RIOT_API_BASE_URL + GET_PUUID_PATH + summonerName + "/" + tagLine + "?api_key=" + API_KEY
	response = requests.get(url)
	url = LOL_API_BASE_URL + GET_ACCOUNT_PATH + response.json()["puuid"] + "?api_key=" + API_KEY
	response = requests.get(url)
	return response.json()["accountId"]

if __name__ == "__main__":
	# Grab 1 person's recent matches
	seed = pull_encrypted_account_id("Nubrozaref", "NA1")

	pull_players_matches(seed)

# Important match data:
# gameId (long)
# gameDuration (long)	duration in seconds
# queueId (int)			Game constant (for filtering)
# teams:
# 	[
#	teamId (int)		100 -> blue side, 200 -> red side
#	win (string)		Fail, Win
# 	]
# participantIdentities:
# 	[
#	participantId (int)		for mapping participant info to account
#	player:
#		[
#		accountId (string)			for tracking unique accounts
#		]
# 	]
# participants:
# 	[
#	participantId (int)		for mapping participant info to account
#	teamId (int)				100 -> blue side, 200 -> red side
#	championId (int)
#	timeline:
#		[
#		role:			DUO, NONE, SOLO, DUO_CARRY, DUO_SUPPORT		
#		lane:			TOP, JUNGLE, MIDDLE, BOTTOM
'''
		Simple position mappings:
		(MIDDLE, SOLO):	MIDDLE
		(TOP, SOLO): TOP
		(JUNGLE, NONE): JUNGLE
		(BOTTOM, DUO_CARRY): BOTTOM
		(BOTTOM, DUO_SUPPORT): SUPPORT
'''
#		]
# 	]