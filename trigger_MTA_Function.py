#########################################################
# This triggers the firebase cloud function every x min #
########################################################

import datetime
import pytz
import time
import urllib2

TIME_TO_RUN = 5 #run every 5 min
DAYS_TO_RUN = 1
ALLOWABLE   = {}
URL 		= "https://us-central1-lirr-scraping.cloudfunctions.net/mta_delay_grabber"
TOTAL_RUNS  = (60*(DAYS_TO_RUN * 24))//TIME_TO_RUN
already_run = False


for i in range(0,60):
	if i % TIME_TO_RUN == 0:
		ALLOWABLE[i] = None

for _ in range(0,TOTAL_RUNS):
	MINUTE = -1
	while MINUTE not in ALLOWABLE:
		time.sleep(40)
		already_run = False
		MINUTE = datetime.datetime.now().minute
		print(MINUTE)

	tries = 0
	while tries < 5:
		try:
			if already_run == False:
				urllib2.urlopen(URL)
				time.sleep(10)
				already_run = True
			break
		except urllib2.HTTPError:
			print("Couldn't run, trying again")
			time.sleep(5)
			tries += 1




