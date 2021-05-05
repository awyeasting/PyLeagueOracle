import requests

VERSION_INFO_PATH = "http://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_INFO_PATH = "http://ddragon.leagueoflegends.com/cdn/{}/data/en_US/champion.json"

def getRecentGameVersion():
	response = requests.get(VERSION_INFO_PATH)
	js = response.json()
	print("Version:",js[0])
	return js[0]

def getChampionMap(version):
	response = requests.get(CHAMPION_INFO_PATH.format(version))
	js = response.json()["data"]
	ch_map = {}
	for champion in js.items():
		ch_map[int(champion[1]["key"])] = champion[0]
	return ch_map

def getInternalChampMap(champion_map):
	m = {}
	champs = list(champion_map.items())
	champs.sort(key=(lambda x: x[0]))
	for i in range(len(champs)):
		m[champs[i][0]] = i
	print("N champs:", len(champs))
	return m, len(champs)

def invertMap(m):
	nm = {}
	for i in m.items():
		nm[i[1]] = i[0]
		if __name__ == "__main__":
			print("Champ: {},\tid:{}".format(i[1],i[0]))
	return nm

# Map league champion id to champion name
champion_map = getChampionMap(getRecentGameVersion())
# Map champion name to league champion id
champion_name_map = invertMap(champion_map)
# Map league champion id to internal champion id (0 - N champs)
internal_champion_map, n_champs = getInternalChampMap(champion_map)

if __name__ == "__main__":
	print(champion_map)
	print(internal_champion_map)