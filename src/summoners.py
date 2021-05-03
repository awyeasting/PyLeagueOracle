import requests
import time

from HIDDEN_CONFIG import API_KEY
from config import LOL_API_BASE_URL, TIER_MAPPING, RATE_LIMIT

# process for getting challenger account ids:
# /lol/league/v4/challengerleagues/by-queue/{queue} (queue = "RANKED_SOLO_5x5") -> ["entries"][i]["summonerId"]
GET_CHALLENGER_LEAGUE_PATH = "/lol/league/v4/challengerleagues/by-queue/"
CHALLENGER_QUEUE = "RANKED_SOLO_5x5"
# /lol/summoner/v4/summoners/{encryptedSummonerId} -> ["accountId"]
GET_SUMMONER_PATH = "/lol/summoner/v4/summoners/"

# /lol/league/v4/entries/by-summoner/{encryptedSummonerId} -> ["tier"] ["rank"] ["leaguePoints"]
GET_SUMMONER_RANK_PATH = "/lol/league/v4/entries/by-summoner/"

# Get the current rank of a particular summoner id
def get_summoner_rank(summonerId):
	print("Getting summoner rank for match labeling...", flush=True)
	url = LOL_API_BASE_URL + GET_SUMMONER_RANK_PATH + summonerId + "?api_key=" + API_KEY
	response = requests.get(url)

	# If the summonerId isn't found anymore (moved servers?)
	if response.status_code == 404:
		print("Could not get rank from summonerId")
		return None

	# Try again in 10 * RATE_LIMIT seconds if it didn't go through
	if response.status_code != 200:
		print(response.text)
		print("Get summoner rank filed with response code:", response.status_code)
		time.sleep(10 * RATE_LIMIT)
		return get_summoner_rank(summonerId)

	rank = {}
	queueEntries = response.json()
	for queueEntry in queueEntries:
		if queueEntry["queueType"] == CHALLENGER_QUEUE:
			rank["tier"] = queueEntry["tier"]
			rank["rank"] = queueEntry["rank"]
			rank["leaguePoints"] = queueEntry["leaguePoints"]
			rank["rankMapping"] = TIER_MAPPING.get(rank["tier"], -1)
			break

	print("Found summoner rank:", rank)
	return rank

# Get the account id associated with a particular summoner id
def get_summoner_account_id(summonerId):
	url = LOL_API_BASE_URL + GET_SUMMONER_PATH + summonerId + "?api_key=" + API_KEY
	response = requests.get(url)

	# Try again in 10 * RATE_LIMIT seconds if it didn't go through
	if response.status_code != 200:
		print(response.text)
		print("Get summoner account id failed with code:", response.status_code, flush=True)
		time.sleep(10 * RATE_LIMIT)
		return get_summoner_account_id(summonerId)

	return response.json()["accountId"]

# Get list of summoner ids with associated lp for each summoner
def get_challenger_summoners():
	url = LOL_API_BASE_URL + GET_CHALLENGER_LEAGUE_PATH + CHALLENGER_QUEUE + "?api_key=" + API_KEY
	response = requests.get(url)

	# Try again in 10 * RATE_LIMIT seconds if it didn't go through
	if response.status_code != 200:
		print(response.text)
		print("Get challenger summoners request failed with code:", response.status_code, flush=True)
		time.sleep(10 * RATE_LIMIT)
		return get_challenger_summoners()

	summonerIds = []
	for entry in response.json()["entries"]:
		summonerIds.append((entry["leaguePoints"], entry["summonerId"]))
	return summonerIds

# Get challenger accountId, summonerId pairs to use for match seeding
# Get min(n_summoners, max summoners) - offset pairs if possible
# Otherwise just get min(n_summoners, max summoners) pairs if possible
# Otherise just get max summoners pairs
def get_challenger_seeds(n_summoners,highest_first = False, offset = 0):
	print("Getting challenger seeds...", flush=True)
	summoners = get_challenger_summoners()
	time.sleep(RATE_LIMIT)

	# Sort it to either be descending or ascending (default)
	summoners.sort(key=lambda x: x[0], reverse=highest_first)
	if n_summoners == -1:
		n_summoners = len(summoners)
	elif n_summoners > len(summoners):
		n_summoners = len(summoners)

	if offset > n_summoners:
		offset = 0

	# Look up the account ids of n_summoner summoner ids
	print("Looking up {} summoner account ids...".format(n_summoners), flush=True)
	accounts = []
	for i in range(offset, n_summoners):
		print("Looking up account id {} of {}...".format(i+1, n_summoners), flush=True)
		accountId = get_summoner_account_id(summoners[i][1])
		accounts.append((accountId, summoners[i][1]))
		time.sleep(RATE_LIMIT)

	return accounts