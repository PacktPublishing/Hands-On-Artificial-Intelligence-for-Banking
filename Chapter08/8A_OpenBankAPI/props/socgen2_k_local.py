# This is just for socgen-k test, you make sure Socgen-k server is working well.
# https://socgen-k-api.openbankproject.com/

# API server URL
BASE_URL = "http://127.0.0.1:8080"
API_VERSION = "v2.0.0"
API_VERSION_V210 = "v2.1.0"
# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI = 'http://127.0.0.1/cb'

# login user: 
USERNAME = '1000203892'
PASSWORD = 'fffffffffffffffff'
CONSUMER_KEY = 'gmtcx4letf2isej1slxhpphtnt2jkt30ldazvkmd'

# fromAccount info:
FROM_BANK_ID = '00100'
FROM_ACCOUNT_ID = '410ad4eb-9f63-300f-8cb9-12f0ab677521'

TO_BANK_ID = '00100'
TO_ACCOUNT_ID = '410ad4eb-9f63-300f-8cb9-12f0ab677521'

# Our currency to use
OUR_CURRENCY = 'XAF'

# Our value to transfer
OUR_VALUE = '10'
OUR_VALUE_LARGE = '1001.00'
