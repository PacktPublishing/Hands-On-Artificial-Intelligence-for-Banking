# This is just for socgen-k test, you make sure Socgen-k server is working well.
# https://socgen-k-api.openbankproject.com/

# API server URL
BASE_URL = "https://socgen-k-api.openbankproject.com"
API_VERSION = "v2.0.0"
API_VERSION_V210 = "v2.1.0"
# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI = 'http://127.0.0.1/cb'

# login user: 
USERNAME = '1000203893'
PASSWORD = '123456'
CONSUMER_KEY = '45wpocdzh2uwnorvrk2sfy1rnwyc0h2ff3kdkr2s'

# fromAccount info: 1000203893
FROM_BANK_ID = '00100'
# FROM_ACCOUNT_ID     = '3806441b-bbdf-3c60-b2b3-14e2f645635f' # 0 transaction
FROM_ACCOUNT_ID = '83b96bb4-ae2c-3e90-ad2c-8ce0b4b0023b'  # 3 transactions
# FROM_ACCOUNT_ID = 'df88925b-4a7f-31f6-a077-3dcbd60b669f'  # 12 transaction

TO_BANK_ID = '00100'

# the following there acounds are all belong to login user : 1000203893
# TO_ACCOUNT_ID = '3806441b-bbdf-3c60-b2b3-14e2f645635f'
# TO_ACCOUNT_ID = '83b96bb4-ae2c-3e90-ad2c-8ce0b4b0023b'
# TO_ACCOUNT_ID = 'df88925b-4a7f-31f6-a077-3dcbd60b669f'

# this account is belong to user: 1000203892
TO_ACCOUNT_ID = '410ad4eb-9f63-300f-8cb9-12f0ab677521'

# these accounts are belong to user: 1000203891
# TO_ACCOUNT_ID = '1f5587fa-8ad8-3c6b-8fac-ac3db5bdc3db'

# these accounts are belong to user: 1000203899 --Ulrich Standalone account
# TO_ACCOUNT_ID = 'bb912420-484d-38c2-8c5b-d9772dd5bfbc'
# TO_ACCOUNT_ID = '0796d146-e39c-36a1-85cd-ef74f5d8227d'

# toCountery
# {
#     "name": "test2",
#     "created_by_user_id": "b9ed3a54-1e98-4ca1-9f95-76815373d9f4",
#     "this_bank_id": "00100",
#     "this_account_id": "83b96bb4-ae2c-3e90-ad2c-8ce0b4b0023b",
#     "this_view_id": "owner",
#     "counterparty_id": "a78dab15-1c51-4e1e-bfc2-aa270a60eb6d",
#     "other_bank_routing_scheme": "Agence",
#     "other_account_routing_scheme": "BKCOM_ACCOUNT",
#     "other_bank_routing_address": "00100",
#     "other_account_routing_address": "1000203892",
#     "is_beneficiary": true
# }

# Our currency to use
OUR_CURRENCY = 'XAF'

# Our value to transfer
OUR_VALUE = '10'
OUR_VALUE_LARGE = '1001.00'
