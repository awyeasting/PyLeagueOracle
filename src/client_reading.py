import requests
import base64
import time

import training
import training_config

from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

LOCKFILE_PATH = "B:\\Program Files (x86)\\RiotGames\\Riot Games\\League of Legends\\lockfile"

def getLockfileInfo():
	# Get port number from lockfile 
	f = open(LOCKFILE_PATH, "r")
	lockfileContents = f.read().split(":")
	l = {}
	l["port"] = lockfileContents[2]
	l["user"] = "riot"
	l["pass"] = lockfileContents[3]
	return l 

def queryChampSelect(lfc):
	url = "https://127.0.0.1:{}/lol-champ-select/v1/session".format(lfc["port"])
	response = requests.get(url,headers={'Accept':'application/json'},auth=requests.auth.HTTPBasicAuth(lfc["user"], lfc["pass"]),verify=False)
	return response

def waitForChampSelect(lfc):
	response = queryChampSelect(lfc)
	hadToWait = False
	while response.status_code == 404:
		print("Waiting for champ select...", flush=True)
		time.sleep(5)
		hadToWait = True
		response = queryChampSelect(lfc)
	print("Champ select started!")
	return response, hadToWait

def notFinalSelectPhase(js):
	return js["timer"]["phase"] != "FINALIZATION"

def waitForChampsFinalized(lfc):
	response = queryChampSelect(lfc)
	js = response.json()
	while response.status_code == 200:
		if not notFinalSelectPhase(js):
			break
		print("Waiting for champs to be finalized...", flush=True)
		time.sleep(1)
		response = queryChampSelect(lfc)
		js = response.json()
	if response.status_code == 404:
		print("Champ select ended prematurely")
	else:
		print("Champs all finalized!")
	return response

def waitForChampSelectToEnd(lfc):
	response = queryChampSelect(lfc)
	while response.status_code != 404:
		print("Waiting for champ select to end...")
		time.sleep(1)
		response = queryChampSelect(lfc)
	return response

def readUntilFinalThenPredict(lfc):
	model = training.create_training_model()
	model.load_weights(training_config.SAVE_PATH)

	justGotPrediction = False
	while True:
		response, hadToWait = waitForChampSelect(lfc)
		response = waitForChampsFinalized(lfc)
		if response.status_code == 200:
			game = encodeGameDocument(response.json())
			training.predict_game(model, game, display=True)
			justGotPrediction = True

			waitForChampSelectToEnd(lfc)

def encodeGameDocument(js):
	players = []
	for player in (js["myTeam"] + js["theirTeam"]):
		p = {}
		p["team"] = player["team"] - 1
		p["champion"] = player["championId"]
		players.append(p)
	#for player in js["theirTeam"]:
	#	player["team"] -= 1
	#	player["champion"] = player["championId"]
	#	players.append(player)

	game = {}
	game["players"] = players
	return game

if __name__ == "__main__":

	lfc = getLockfileInfo()
	
	readUntilFinalThenPredict(lfc)

'''from PIL import ImageGrab, Image
import win32gui

def grab_client_img(save_loc="", show_client=False):
	toplist, winlist = [], []
	def enum_cb(hwnd, results):
	    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
	win32gui.EnumWindows(enum_cb, toplist)

	lol = [(hwnd, title) for hwnd, title in winlist if 'league of legends' in title.lower()]
	# just grab the hwnd for first window matching firefox
	lol = lol[0]
	hwnd = lol[0]

	win32gui.SetForegroundWindow(hwnd)
	bbox = win32gui.GetWindowRect(hwnd)
	img = ImageGrab.grab(bbox)
	if show_client:
		img.show()
	if save_loc:
		img.save(save_loc)

	return img

def get_player_portraits(img):
	team1Portraits = []
	team2Portraits = []
	for i in range(5):
		portrait = img.crop((10,119 + 100 * i,396,119 + 100 * (i+1)))
		team1Portraits.append(portrait)
	for i in range(5):
		portrait = img.crop((1600-396,119 + 100 * i,1600-10,119 + 100 * (i+1)))
		team2Portraits.append(portrait)
	return team1Portraits, team2Portraits

if __name__ == "__main__":
	#img = grab_client_img(save_loc="client.png", show_client=True)
	img = Image.open("clientpregame.png")
	team1Portraits, team2Portraits = get_player_portraits
	print(img)
'''