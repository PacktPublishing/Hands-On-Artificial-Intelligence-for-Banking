# -*- coding: utf-8 -*-

from __future__ import print_function    # (at top of module)
import sys, requests


# Note: in order to use this example, you need to have at least one account
# that you can send money from (i.e. be the owner).
# All properties are now kept in one central place

from props.default import *

# You probably don't need to change those
import lib.obp
obp = lib.obp

obp.setBaseUrl(BASE_URL)
obp.setApiVersion(API_VERSION)

#add the followings
CONSUMER_KEY  = '<ENTER YOUR CONSUMER_KEY>'
USERNAME = '<ENTER YOUR USERNAME>'
PASSWORD = '<ENTER YOUR PASSWORD>'
OUR_BANK = '<ENTER YOUR BANK>'
#rbs

# login and set authorized token
obp.login(USERNAME, PASSWORD, CONSUMER_KEY)
obp.setCounterParty(COUNTERPARTY_BANK, OUR_COUNTERPARTY,OUR_COUNTERPARTY_ID,OUR_COUNTERPARTY_IBAN)
obp.setPaymentDetails(OUR_CURRENCY, OUR_VALUE_LARGE)
banks = obp.getBanks()

our_bank = banks[0]['id']

#OUR_BANK = 'dmo.01.uk.uk'
our_bank = OUR_BANK

cp_bank = obp.getCounterBankId()
cp_account = obp.getCounterpartyAccountId()

print ("our bank: {0}".format(our_bank))

#get accounts for a specific bank
print (" --- Private accounts")

accounts = obp.getPrivateAccounts(our_bank)

for a in accounts:
    print (a['id'])

#just picking first account
our_account = accounts[0]['id']

print ("our account: {0}".format(our_account))

print ("")
print (" --- Get owner transactions")
transactions = obp.getTransactions(our_bank, our_account) 
print ("Got {0} transactions".format(len(transactions)))

print (" --- Get challenge request types")
challenge_types = obp.getChallengeTypes(our_bank, our_account) 
print (challenge_types)
challenge_type = challenge_types[0]
print (challenge_type)

print ("")
print ("Initiate transaction request (small value)")
initiate_response = obp.initiateTransactionRequest(our_bank, our_account, challenge_type, cp_bank, cp_account) 

if "error" in initiate_response:
    sys.exit("Got an error: " + str(initiate_response))

if (initiate_response['challenge'] != None):
    #we need to answer the challenge
    challenge_query = initiate_response['challenge']['id']
    #transaction_req_id = initiate_response['id']['value']
    transaction_req_id = initiate_response['id']

    challenge_response = obp.answerChallenge(our_bank, our_account, transaction_req_id, challenge_query) 
    if "error" in challenge_response:
        sys.exit("Got an error: " + str(challenge_response))

    print ("Transaction status: {0}".format(challenge_response['status']))
    print ("Transaction created: {0}".format(challenge_response["transaction_ids"]))
else:
    #There was no challenge, transaction was created immediately
    print ("Transaction was successfully created:")
    print ("{0}".format(initiate_response))
#There was no challenge, transaction was created immediately
print ("Transaction was successfully created:")
print ("{0}".format(initiate_response))


#add the following lines to hello-obp.py before running it
#add lines to download the file
print("")
print(" --- export json")
import json
f_json = open('transactions.json','w+')
json.dump(initiate_response,f_json,sort_keys=True, indent=4)
f_json.close()
