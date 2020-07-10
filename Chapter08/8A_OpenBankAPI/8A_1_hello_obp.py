# -*- coding: utf-8 -*-

from __future__ import print_function    # (at top of module)
import sys
import time
import requests


# Note: in order to use this example, you need to have at least one account
# that you can send money from (i.e. be the owner).
# All properties are now kept in one central place

from props.default import *


# You probably don't need to change those
import lib.obp
obp = lib.obp

obp.setBaseUrl(BASE_URL)
obp.setApiVersion(API_VERSION)

# Login and set authorized token
obp.login(USERNAME, PASSWORD, CONSUMER_KEY)

# Get current user
print("")
print(" --- Get current user")
user = obp.getCurrentUser()
print("current user data:\n{0}".format(user))
user_id = user['user_id']
print("current user id: {0}".format(user))

our_bank = OUR_BANK  # banks[0]['id']
print("our bank: {0}".format(our_bank))

# Get accounts for a specific bank
print(" --- Private accounts")

accounts = obp.getPrivateAccounts(our_bank)

for a in accounts:
    print(a['id'])

# Just picking first account
our_account = accounts[0]['id']
print("our account: {0}".format(our_account))

# Reload account
print("")
print(" --- Load our account data")
account_data = obp.getAccount(our_bank, our_account)
print("our account data:\n{0}".format(account_data))

print("")
print(" --- Modify account label")
new_label = "New label %s" % time.strftime("%d/%m/%Y %I:%M:%S")
print(new_label)

# Prepare post data and set new label value
post_data = {
    'id': '%s' % our_account,
    'label': '%s' % new_label,
    'bank_id': '%s' % OUR_BANK
}

# Send post request with attached json with new label value
response = requests.post(
    u"{0}/obp/{1}/banks/{2}/accounts/{3}".format(
        BASE_URL, API_VERSION, our_bank, our_account),
    json=post_data,
    headers=obp.mergeHeaders(obp.DL_TOKEN, obp.CONTENT_JSON)
)

# Print result
print("")
print(response.status_code)
print(response.text)

# Reload account again for comparison
account = obp.getAccount(our_bank, our_account)
print("")
print(" --- Reload account data")
print("our account data after label update:\n{0}".format(account))


print("")
print(" --- Get owner transactions")
transactions = obp.getTransactions(our_bank, our_account)
print("Got {0} transactions".format(len(transactions)))
for t in transactions:
    print('{0} ({1} {2})'.format(
        t['id'],
        t['details']['value']['currency'],
        t['details']['value']['amount'])
    )

#add the following lines to hello-obp.py before running it
#add lines to download the file
print("")
print(" --- export json")
import json
f_json = open('transactions.json','w+')
json.dump(transactions,f_json,sort_keys=True, indent=4)
