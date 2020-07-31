# -*- coding: utf-8 -*-
"""
Settings for the hello scripts.

You most likely need to edit a few of them, e.g. API_HOST and the OAuth
credentials.
"""

OUR_BANK         = 'dan.01.uk.uk'

USERNAME         = 'Robert.Uk.01'
PASSWORD         = '356609'
CONSUMER_KEY     = 'fz3as2gadnzustw5sbnokwspqnit4obdwpsowuif'
CONSUMER_SECRET  = 'lqh3owrxqnxkkuaq3yrickxh433nzv3uhze1qc1e'

# API server URL
BASE_URL         = "https://danskebank.openbankproject.com"
API_VERSION      = "v2.0.0"

# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI     = 'http://127.0.0.1/cb'

# Our COUNTERPARTY account id (of the same currency)
OUR_COUNTERPARTY  = 'be4c3b50-fa7a-4e38-989b-e6d3c874a368'
COUNTERPARTY_BANK = 'dan.01.uk.uk'
# this following two fields are just used in V210
OUR_COUNTERPARTY_ID = ''
OUR_COUNTERPARTY_IBAN = ''

# Our currency to use
OUR_CURRENCY     = 'GBP'

# Our value to transfer
# values below 1000 do not requre challenge request
OUR_VALUE = '0.01'
OUR_VALUE_LARGE  = '1000.00'
PAYMENT_DESCRIPTION = 'Hello Payments v2.1!'

