# --------------------------------------
#            INTERNAL CONFIG
# --------------------------------------

TIER_MAPPING = {
	"CHALLENGER": 0,
	"GRANDMASTER": 1,
	"MASTER": 2,
	"DIAMOND": 3,
	"PLATINUM": 4,
	"GOLD": 5,
	"SILVER": 6,
	"BRONZE": 7,
	"IRON": 8
}

# How old of matches should be considered
MAX_MATCH_AGE_DAYS = 30
# Whether or not to attach match seed's rank to game
PULL_SUMMONER_RANK = True

# --------------------------------------
#            DATABASE CONFIG
# --------------------------------------

LOL_DB_NAME = "leagueoflegends"
MATCH_COL_NAME = "matches-v2"

# --------------------------------------
#              API CONFIG
# --------------------------------------

# Keys expire in 24 hours
# 20 requests / 1 second
# 100 requests / 2 minutes
RIOT_API_BASE_URL = "https://americas.api.riotgames.com"
LOL_API_BASE_URL = "https://na1.api.riotgames.com"
# Currently rate limited to 120 seconds / 100 matches
# Pad rate limit by 5% to try to ensure no problems with too many requests
RATE_LIMIT = (120 / 100) * 1.01
# match-v4: /lol/match/v4/matches/{matchId}		Get match by match ID
# match-v4: /lol/match/v4/matchlists/by-account/{encryptedAccountId}

# I'm not kidding, this is actually Riot game's constant for the ranked solo/duo queue games
RANKED_QUEUE_ID = 420

# API NOTES:
# Three ids: summoner id, account id, puuid
# summoner and account ids are unique per region
# puuids are unique globally