import time

from config import RATE_LIMIT

def waitForRequestTime(last_request_time):
	wait = RATE_LIMIT - (time.time()-last_request_time)
	if wait > 0:
		time.sleep(wait)