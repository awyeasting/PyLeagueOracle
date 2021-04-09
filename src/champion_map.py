import requests

CHAMPION_INFO_PATH = "http://ddragon.leagueoflegends.com/cdn/11.7.1/data/en_US/champion.json"

def getChampionMap():
	response = requests.get(CHAMPION_INFO_PATH)
	js = response.json()["data"]
	ch_map = {}
	for champion in js.items():
		ch_map[int(champion[1]["key"])] = champion[0]
	return ch_map

def getInternalChampMap(champion_map):
	m = {}
	champs = champion_map.items().sort(key=(lambda x: x[0]))
	for i in range(len(champs)):
		m[champs[i][0]] = i
	return m

# Map league champion id to champion name
champion_map = getChampionMap()
# Map league champion id to internal champion id (0 - N champs)
internal_champion_map = getInternalChampMap(champion_map)