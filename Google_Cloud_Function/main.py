def hello_remote_config(request):
	
	import requests
	import time
	import datetime
	from pytz import timezone
	import json
	import gspread
	import google.auth
	from oauth2client.service_account import ServiceAccountCredentials
	from google.auth.transport.requests import AuthorizedSession

	test 	  	= requests.get("https://collector-otp-prod.camsys-apps.com/realtime/serviceStatus?apikey=API_KEY")
	parsed 	  	= json.loads(test.content)
	scope 		= ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
	credentials 	= ServiceAccountCredentials.from_json_keyfile_name('client_secret.json',scope)
	gc 		= gspread.authorize(credentials)
	worksheet 	= gc.open("MTA_DELAY_DATA_DUMP").sheet1
	tz 		= timezone('America/New_York')

	'Sometimes duplicates appear. This is to make sure we dont add more than we need'				
	already_seen 	 = {}
	converter 	 = {
					"2":0.0,"3":0.1,"1":0.2,"S":0.3,"W":0.4,"SIR":0.5,
					"Z":0.6,"N":0.7,"L":0.8,"M":0.9,"R":0.10,"Q":0.11,
					"F":0.12,"G":0.13,"D":0.14,"E":0.15,"J":0.16,"S":0.17,
					"B":0.18,"C":0.19,"A":0.20,"6":0.21,"7":0.22,"4":0.23,"5":0.24}


	for i in range(330,370):

		train_line 	= parsed['routeDetails'][i]['route']
		vacant_cell	= int(worksheet.acell('F2').value)

		if train_line in converter:
		
			train_line = converter[train_line]
		
			if train_line not in already_seen:
			
				try:

					value = parsed['routeDetails'][i]['statusDetails'][0]['statusSummary']

					if value == 'Delays':
						compiled = str(datetime.datetime.now(tz).time()) + "|" + str(train_line) + "|" + "0.1" + "|" + str(datetime.datetime.now(tz).today())
						worksheet.update_cell(vacant_cell,1,compiled)
						worksheet.update_acell('F2',vacant_cell+1)

					else:
						compiled = str(datetime.datetime.now(tz).time()) + "|" + str(train_line) + "|" + "0.0" + "|" + str(datetime.datetime.now(tz).today())
						#worksheet.update_cell(vacant_cell,1,compiled)
						#worksheet.update_acell('F2',vacant_cell+1)

					already_seen[train_line] = None

				except KeyError:
						compiled = str(datetime.datetime.now(tz).time()) + "|" + str(train_line) + "|" + "0.0" + "|" + str(datetime.datetime.now(tz).today())
						#worksheet.update_cell(vacant_cell,1,compiled)
						#worksheet.update_acell('F2',vacant_cell+1)
						#already_seen[train_line] = None




