# API server URL
BASE_URL = "https://apisandbox.openbankproject.com"
API_VERSION = "v2.0.0"
API_VERSION_V210 = "v2.1.0"
API_VERSION_V220 = "v2.2.0"
# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI = 'https://apisandbox.openbankproject.com/cb'

# login user: 
USERNAME = 'susan.uk.29@example.com'
PASSWORD = '2b78e8'
CONSUMER_KEY = 'mmsjy5gv3ha1achrnqfrea4u42gxi5wsowownpfb'
#Consumer Secret lez2xf3jiz1quhxxhrmdr200hpxdwjwybjmeidjd
#Consumer ID 2193

# fromAccount info:
FROM_BANK_ID = 'gh.29.uk'
FROM_ACCOUNT_ID = '8ca8a7e4-6d02-48e3-a029-0b2bf89de9f0'

# toBankAccount and toCounterparty info(These data is from kafka side): 
TO_BANK_ID = 'gh.29.uk'
TO_ACCOUNT_ID = '851273ba-90d5-43d7-bb31-ea8eba5903c7'
# TO_COUNTERPARTY_ID = 'a635f6ff-c26b-46ad-8194-2406bacceae4'
# TO_COUNTERPARTY_IBAN = 'DE12 1234 5123 4510 2207 8077 877'

# Our currency to use
OUR_CURRENCY = 'GBP'
# Our value to transfer
# values below 1000 do not require challenge request
OUR_VALUE = '1.00'
OUR_VALUE_LARGE = '1001.00'
