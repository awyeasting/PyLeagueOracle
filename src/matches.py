import requests
import time

from HIDDEN_CONFIG import API_KEY
from config import LOL_API_BASE_URL, RATE_LIMIT, MAX_MATCH_AGE_DAYS
from summoners import get_challenger_seeds, get_summoner_rank
from accounts import get_player_match_info

# /lol/match/v4/matches/{matchId} -> match
GET_MATCH_PATH = "/lol/match/v4/matches/"

# Pull the relevant data from a single match for storage
def get_match_data(matchId):
	match_data = {}
	url = LOL_API_BASE_URL + GET_MATCH_PATH + str(matchId) + "?api_key=" + API_KEY
	response = requests.get(url)
	match = response.json()

	# If the matchId isn't found anymore (too old now?)
	if response.status_code == 404:
		print("Match id not found, skipping seed")
		return None

	# Try again in 10 * RATE_LIMIT seconds if it didn't go through
	if response.status_code != 200:
		print(response.text)
		print("Pull match request failed with code:",response.status_code)
		time.sleep(10 * RATE_LIMIT)
		return get_match_data(matchId)

	# Get player non specific data
	match_data["gameId"] = matchId
	match_data["gameCreation"] = match["gameCreation"]
	match_data["gameDuration"] = match["gameDuration"]

	# Get player specific data
	players = []
	for pIden in match["participantIdentities"]:
		player = {}
		player["id"] = pIden["participantId"]
		p = match["participants"][player["id"] - 1]
		player["team"] = int(200 == p["teamId"])
		player["champion"] = p["championId"]
		player["role"] = p["timeline"]["role"]
		player["lane"] = p["timeline"]["lane"]
		player["stats"] = p["stats"]
		
		player["accountId"] = pIden["player"]["accountId"]
		player["summonerId"] = pIden["player"]["summonerId"]
		players.append(player)
	match_data["players"] = players

	# Get match outcome
	outcome = match["teams"][0]["win"] == "Fail"
	if match["teams"][0]["teamId"] == 200:
		outcome = not outcome
	match_data["outcome"] = outcome

	# Print match data for debugging purposes (must import from pulldata.py)
	#print_match(players, outcome)
	return match_data

# Do a breadth first graph traversal of league of legends matches to search for unseen
# match data to save
def pull_many_matches(seeds=[], matchCol=None):
	if matchCol == None:
		print("Must initialize mongodb match collection before pulling many matches")
		return

	n_matches_added = 0
	seeds_looked_at = 0
	last_n_seeds = 0
	n_seeds = 1
	start_time = time.time()
	while True:
		# If there are currently no seeds then get the challenger seeds
		if not len(seeds):
			# Offset for skipping past potentially recently seen seeds
			# (function guarantees that it will be able to return seeds if offset gets too high)
			seeds = get_challenger_seeds(n_seeds, offset = last_n_seeds)
			
			# Increase the number of seed accounts to look at next time if the seeds run out 
			# (likely to only happen in cases where there aren't recent enough games on the lowest ranked challenger summoners)
			last_n_seeds = n_seeds
			n_seeds *= 2

		# Get first seed in queue
		seed = seeds.pop(0)
		seeds_looked_at += 1

		# Pull player match info
		player_matches = get_player_match_info(seed[0])
		oldestTimeStamp = 1000*(time.time() - (MAX_MATCH_AGE_DAYS * 24 * 60 * 60))
		time.sleep(RATE_LIMIT)

		# If player account not found, then skip this seed
		if player_matches == None:
			print("Skipping seed with no matches...")
			seeds_looked_at -= 1
			continue

		# Empty seed rank so it knows to get it for the new seed before saving a match
		seed_rank = None

		# Pull match data
		for match in player_matches:
			if match["timestamp"] < oldestTimeStamp:
				print("No more new enough matches on current player", flush=True)
				break

			# Check that match is not already in database
			if matchCol.count_documents({'gameId': match["gameId"]}) == 0:
				
				print("\nPulling new match...", flush=True)

				# Pull match data and add to database it if it isn't
				try:
					match_data = get_match_data(match["gameId"])
				except KeyError:
					print("Unknown error occurred while trying to pull match data, skipping...")
					continue
				time.sleep(RATE_LIMIT)

				# If no match found for the id then check next match
				if match_data == None:
					continue

				# Pull players from match and add to seeds list (if they're not already there)
				print("Stripping players from match...", flush=True)
				i = 0
				seed_accounts = list(map(lambda x: x[0], seeds))
				for player in match_data["players"]:
					if (player["accountId"] not in seed_accounts) and (player["accountId"] != seed[0]):
						seeds.append((player["accountId"], player["summonerId"]))
						i+=1
				print("Added {} players to seed players queue ({} players in queue)".format(i,len(seeds)), flush=True)

				# Attach seed's rank to match
				if seed_rank == None:
					seed_rank = get_summoner_rank(seed[1])
					time.sleep(RATE_LIMIT)
				match_data["seed_rank"] = seed_rank

				# Put into database
				matchCol.insert_one(match_data)
				n_matches_added+=1
				cur_time = time.time()
				print("Inserted match into database ({} matches added from {} seeds)".format(n_matches_added, seeds_looked_at))
				matches_per_second = n_matches_added / (cur_time - start_time)
				print("{:.2f} matches per second ({:.2f} matches per minutes)".format(matches_per_second, matches_per_second * 60), flush=True)
