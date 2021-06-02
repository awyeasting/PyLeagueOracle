import requests
import time

from HIDDEN_CONFIG import API_KEY
from config import RIOT_API_BASE_URL, LOL_API_BASE_URL, RATE_LIMIT
from util import waitApiRequest, waitApiClear

# process for getting encrypted account id
# /riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine} -> encrypted PUUID
GET_PUUID_PATH = "/riot/account/v1/accounts/by-riot-id/"
# /lol/summoner/v4/summoners/by-puuid/{encryptedPUUID} -> encrypted Account ID
GET_ACCOUNT_PATH = "/lol/summoner/v4/summoners/by-puuid/"
# Nubrozaref#NA1 current encrypted puuid:	"6Uf6wy09tg40GDDz6VQ2WfUi63aDPBuWZMQn4xCzFRo03MvskmHpkHzPlOHfLgmfclV4MItRLoZRvg"
# 				current encrypted account id: "eJwW8oKNquGnNm6CcyxlQpZt6MiRhcApI-i162LOh-rVoA"

# /lol/match/v4/matchlists/by-account/{encryptedAccountId} (queue = 420 for solo queue) -> [gameId]
GET_MATCHES_PATH = "/lol/match/v4/matchlists/by-account/"

def get_encrypted_account_id(summonerName, tagLine):
	waitApiRequest("account","by-riot-id")
	print("Pulling PUUID for from summonerName, tagline pair...", flush=True)
	url = RIOT_API_BASE_URL + GET_PUUID_PATH + summonerName + "/" + tagLine + "?api_key=" + API_KEY
	response = requests.get(url)

	waitApiRequest("summoner","by-puuid")
	print("Pulling encrypted account id from PUUID...", flush=True)
	url = LOL_API_BASE_URL + GET_ACCOUNT_PATH + response.json()["puuid"] + "?api_key=" + API_KEY
	response = requests.get(url)
	return response.json()["accountId"]

# Pull recent ranked matched associated with a specific encryptedAccountId
def get_player_match_info(encryptedAccountId):
	waitApiRequest("match-v4","matchlists")
	print("Getting matches for player with encrypted account id...", flush=True)
	url = LOL_API_BASE_URL + GET_MATCHES_PATH + encryptedAccountId + "?queue=420&api_key=" + API_KEY
	response = requests.get(url)

	# If the accountId isn't found anymore (moved servers?)
	if response.status_code == 404:
		print("Encrypted account id not found, skipping seed")
		return None

	# Try again in 10 * RATE_LIMIT seconds if it didn't go through
	if response.status_code != 200:
		print(response.text)
		print("Pull player match info request failed with code:", response.status_code)
		# Wait for messages to clear if rate limited
		if response.status_code == 429:
			waitApiClear("match-v4","matchlists")
		return get_player_match_info(encryptedAccountId)

	matches = response.json()["matches"]
	print("Found",len(matches),"matches")
	return matches