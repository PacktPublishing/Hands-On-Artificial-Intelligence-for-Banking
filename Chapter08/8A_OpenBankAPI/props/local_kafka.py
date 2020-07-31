# This is just for local test, you need first run the OBP-API locally,
# and you need prepare the USERNAME, PASSWORD and CONSUMER_KEY.
# if you want to transfer to other account or counterparty, you need prepare your own data.

# API server URL
BASE_URL = "http://127.0.0.1:8080"
API_VERSION = "v2.0.0"
API_VERSION_V210 = "v2.1.0"
API_VERSION_V220 = "v2.2.0"
# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI = 'http://127.0.0.1/cb'

# login user: 
USERNAME = 'robert.y.9.gh@example.com'
PASSWORD = 'c9a641df'
CONSUMER_KEY = '5du40pymqlgufkncfxn3cny5zhlxv3kts5d2a3lj'

# fromAccount info:
FROM_BANK_ID = 'obp-bank-y-gh'
FROM_ACCOUNT_ID = 'e4f001fe-0f0d-4f93-a8b2-d865077315ec'

# toBankAccount and toCounterparty info(These data is from kafka side): 
TO_BANK_ID = 'obp-bank-x-gh'
TO_ACCOUNT_ID = 'e4f001fe-0f0d-4f93-a8b2-d865077315ec'
TO_COUNTERPARTY_ID = 'a635f6ff-c26b-46ad-8194-2406bacceae4'
TO_COUNTERPARTY_IBAN = 'DE12 1234 5123 4510 2207 8077 877'

# Our currency to use
OUR_CURRENCY = 'GBP'
# Our value to transfer
# values below 1000 do not require challenge request
OUR_VALUE = '0.01'
OUR_VALUE_LARGE = '1001.00'
