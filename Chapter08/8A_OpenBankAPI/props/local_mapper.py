# This is just for local test, you need first run the OBP-API locally,
# and maybe prepare the CONSUMER_KEY, OUR_COUNTERPARTY_ACCOUNT_ID, OUR_COUNTERPARTY_ID, OUR_COUNTERPARTY_IBAN locally


# API server URL
BASE_URL  = "http://127.0.0.1:8080"
API_VERSION  = "v2.0.0"
API_VERSION_V210  = "v2.1.0"

# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI = 'http://127.0.0.1/cb'


#login user: 
USERNAME     = 'susan.uk.29@example.com'
PASSWORD     = '2b78e8'
CONSUMER_KEY = '3ru5q4x4bfdeicdqpxwrylujtmwfj5r1m2gluaef'

#fromAccount info:
FROM_BANK_ID     = 'gh.29.uk'
FROM_ACCOUNT_ID = '8ca8a7e4-6d02-48e3-a029-0b2bf89de9f0'

#toBankAccount and toCounterparty info: 
TO_BANK_ID = 'gh.29.uk'
TO_ACCOUNT_ID = '26fa77a2-028f-4022-b47d-26caa8db359a'

# Our currency to use
OUR_CURRENCY = 'GBP'

# Our value to transfer
# values below 1000 do not require challenge request
OUR_VALUE = '0.01'
OUR_VALUE_LARGE = '1001.00'
