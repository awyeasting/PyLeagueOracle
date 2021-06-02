import time
import queue

from config import RATE_LIMIT

# 100 requests per 2 minutes
API_LIMIT = [(20,1),(100, 120)]
API_LIMITS = {
	"match-v4": {
		# 500 requests per 10 seconds
		"matches": [(500, 10)],
		# 1000 requests per 10 seconds
		"matchlists": [(1000, 10)]
	},
	"league-v4": {
		# 60 requests every 1 minute
		"by-summoner": [(60, 60)],
		# 500 requests every 10 minutes
		"challengerleagues": [(500, 60*10)]
	},
	"summoner-v4": {
		# 1600 requests every 1 minute
		"summoners": [(1600, 60)],
		# 1600 requests every 1 minute
		"by-puuid": [(1600, 60)]
	}
}
print(API_LIMITS)

def buildMessageQueues():
	queues = {}
	for (api, methods) in API_LIMITS.items():
		queues[api] = {}
		for (method, limits) in methods.items():
			queues[api][method] = [queue.Queue(limit[0]) for limit in limits]
	return queues

API_QUEUE = [queue.Queue(API_LIMIT[0][0]), queue.Queue(API_LIMIT[1][0])]
API_QUEUES = buildMessageQueues()
print(API_QUEUES)

def waitApiRequest(api_name, method_name):
	# Get the message wait queue
	q = API_QUEUES[api_name][method_name]
	# Get the rate limit for the message wait queue
	lim = API_LIMITS[api_name][method_name]
	
	# General wait
	# Only wait if you need to wait (More messages than rate limit)
	for i in range(len(API_QUEUE)):
		if API_QUEUE[i].full():
			# Calculate how old a message needs to be to be irrelevant to the rate limit
			oldMessageTime = time.time() - API_LIMIT[i][1]
			# Wait until the oldest message ages out (if it needs to age out)
			waitTime = API_QUEUE[i].get() - oldMessageTime
			if waitTime > 0:
				time.sleep(waitTime)
	
	# Route specific
	# Only wait if you need to wait (More messages than rate limit)
	for i in range(len(q)):
		if q[i].full():
			# Calculate how old a message needs to be to be irrelevant to the rate limit
			oldMessageTime = time.time() - lim[i][1]
			# Wait until the oldest message ages out (if it needs to age out)
			waitTime = q[i].get() - oldMessageTime
			if waitTime > 0:
				time.sleep(waitTime)

	# Add current time as new message time
	rtime = time.time()
	for i in range(len(q)):
		q[i].put(rtime)
	API_QUEUE[0].put(rtime)
	API_QUEUE[1].put(rtime)

def waitApiClear(api_name, method_name):
	# Get the rate limit for the message wait queue
	lim = API_LIMITS[api_name][method_name]

	# Wait the max time for all messages to age out
	time.sleep(lim[-1][1])

def rateLimitRequest(url, api_name, method_name):
	waitApiRequest(api_name, method_name)
	response = requests.get(url)

	if response.status_code != 200:
		# If not found, then skip!
		if response.status_code == 404:
			print("{}, {}: 404'd, skipping...".format(api_name, method_name))
			return None

		# Otherwise print debug information
		# and try again
		print(response.text)
		print("{}, {}: request failed with code:".format(api_name, method_name), response.status_code)
		# Wait for messages to clear if rate limited
		if response.status_code == 429:
			waitApiClear(api_name, method_name)
		return rateLimitRequest(url, api_name, method_name)