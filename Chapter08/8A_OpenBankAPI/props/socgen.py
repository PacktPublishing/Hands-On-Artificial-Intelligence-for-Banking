# -*- coding: utf-8 -*-
"""
Settings for the hello scripts.

You most likely need to edit a few of them, e.g. API_HOST and the OAuth
credentials.
"""

OUR_BANK         = '00100'

USERNAME         = '1000203893'
PASSWORD         = '1000203893'
CONSUMER_KEY     = 'bvldezvlnqj4mtva4jfktke4xliep0bt1xm44yxi'
CONSUMER_SECRET  = 'fgwo35uhkroebasxlqgzjjcc0cf1yaujuynkwodz'

# API server URL
BASE_URL         = 'https://socgen2-k-api.openbankproject.com'
API_VERSION      = "v2.1.0"

# API server will redirect your browser to this URL, should be non-functional
# You will paste the redirect location here when running the script
CALLBACK_URI     = 'http://127.0.0.1/cb'

# Our COUNTERPARTY account id (of the same currency)
OUR_COUNTERPARTY  = '3806441b-bbdf-3c60-b2b3-14e2f645635f'
COUNTERPARTY_BANK = '00100'
# this following two fields are just used in V210
OUR_COUNTERPARTY_ID = ''
OUR_COUNTERPARTY_IBAN = ''


# Our currency to use
OUR_CURRENCY     = 'XAF'

# Our value to transfer
# values below 1000 do not requre challenge request
OUR_VALUE        = '0.01'
OUR_VALUE_LARGE  = '1000.00'
PAYMENT_DESCRIPTION = 'Hello Payments v2.1!'
