'''*************************************
##Section 1. Import relevant libraries & constants
'''
import datetime
import random
import numpy as np
from scipy.stats import norm
import pandas as pd
import copy
##import keras
from keras.models import Model,load_model
from keras.layers import Dense, Input

'''*************************************
##Section 2. Define program-wide variables and values
'''
simulation_mc = True
num_pricing_points = 15

##Load list of depositor and borrower
f_deposit_path = 'deposit_list.csv'
f_loan_path = 'loan_list.csv'
list_depositors = []
list_borrowers = []
dep_model_path = 'dep_model.h5'
loan_model_path = 'loan_model.h5'
reward_model_path = 'reward_model.h5'
funding_model_path = 'funding_model.h5'
bank_checking_interval = 1
start_date = datetime.datetime(2018,1, 1,0,0)
self_funding_target = 0.0
reward_model_input_features = 15*8 + 7
output_log = '/home/jeff/log.txt'
f_log = open(output_log,"w+")

result_deposit= []
result_expense= []
result_loan= []
result_income= []
result_path = 'output_result.csv'
f_output = open(result_path, "w+")

list_depositors = []
list_borrowers = []

loan_empty_grid = dict([(0,0.0),(1,0.0),(2,0.0),(5,0.0),(7,0.0),(10,0.0),(14,0.0),(28,0.0),(29,0.0),(30,0.0),(31,0.0),(60,0.0),(90,0.0),(180,0.0),(360,0.0)])
deposit_empty_grid = dict([(0,0.0),(1,0.0),(2,0.0),(5,0.0),(7,0.0),(10,0.0),(14,0.0),(28,0.0),(29,0.0),(30,0.0),(31,0.0),(60,0.0),(90,0.0),(180,0.0),(360,0.0)])

deposit_constant_grid = dict([(0,0.04),(1,0.13),(2,0.31),(5,0.37),(7,0.46),(10,0.56),(14,0.74),(28,0.79),(29,0.93),(30,1.06),(31,1.15),(60,1.27),(90,1.27),(180,1.42),(360,1.61)])
loan_constant_grid = dict([(0,0.01),(1,1.01),(2,2.01),(5,5.01),(7,7.01),(10,10.01),(14,14.01),(28,28.01),(29,29.01),(30,30.01),(31,31.01),(60,60.01),(90,90.01),(180,180.01),(360,360.01)])

prev_failed_bank = False
num_randomized_grid = 20

maturing_depositors_amt = deposit_empty_grid
maturing_borrowers_amt = loan_empty_grid

'''*************************************
##Section 3. Define all the classes and functions
3A: helper function for pricing value simulation
3B: depositor class
3C: borrower class
3D: bank class
3E: environment class
3F: function to load loan and deposit as borrowers and depositors
'''
##3A - helper functions for pricing value simulation

##3A.i
#monte carlos simulation for pricing acceptance of clients
def pricing_sensitivity_mc(given_prob, pricing_offered):
    #1 bps is the minimum price
    elasticity = max((norm.ppf(given_prob)/4+1),0.01)
    result = elasticity * pricing_offered
    return result

##3A.ii
#elasticity simulation for pricing acceptance of clients
def pricing_sensitivity_linear(given_prob, pricing_offered):
    #range from 0.5 to 1.5
    elasticity = given_prob + 0.5
    result = given_prob * elasticity
    return result

##3B - depositor
##assume deposit amount won't change
##depositor and borrower are different parties
#class depositor simulate for behavior of one depositor
class depositor(object):

    #3B.i init
    def __init__(self,curr_amt, curr_pric,curr_m, start_dte):
        self.curr_amt = curr_amt
        self.curr_pric = curr_pric
        self.curr_m = curr_m
        self.prob_attrite = random.uniform(0,1)
        self.start_dte = start_dte
        self.m_dte = self.start_dte + datetime.timedelta(days=self.curr_m)
        self.exist = False
        self.attrite_threshold = curr_pric
        return

    #3B.ii set deposit expectation given the market pricing
    def set_deposit_expectation(self,curr_dte,mkt_deposit_pricing_grid):
        if curr_dte == self.m_dte:
            ##MAX_INT = 60
            mkt_pricing = mkt_deposit_pricing_grid[self.curr_m]
            self.prob_attrite = random.uniform(0,1)
            prob_retention = 1 - self.prob_attrite
            pricing_considered = mkt_pricing* 0.95
            #if behavior is based on randomized normal distribution
            if simulation_mc==True:
                attrite_threshold = pricing_sensitivity_mc(prob_retention, pricing_considered)
            #if behavior is based on price elasticity
            else:
                attrite_threshold = pricing_sensitivity_linear(prob_retention, pricing_considered)
            self.attrite_threshold = attrite_threshold
            return self.attrite_threshold, self.curr_m, self.curr_amt
        else:
            #no willingness to go
            self.attrite_threshold = 0
            return -1,self.curr_m, self.curr_amt

    def get_deposit_maturing(self,tomorrow_dte):
        if tomorrow_dte == self.m_dte:
            return self.curr_m, self.curr_amt
        else:
            #no expectation if not yet mature
            return self.curr_m, 0

    #3B.iii make the carry forward or not decision based on the expectation and pricing offered by the bank
    #run at maturity date to determine if the deposit should stay or withdrawn from the bank
    def set_deposit_carry_fwd(self,pricing_dict,curr_dte):
        if self.m_dte <= curr_dte:
            pricing_offered = 0
            try:
                pricing_offered = pricing_dict[self.curr_m]
            except Exception:
                print(self.curr_m)

            self.start_dte = curr_dte
            self.curr_pric = self.attrite_threshold
            self.m_dte = self.start_dte + datetime.timedelta(days=self.curr_m)
            self.prob_attrite = random.uniform(0,1)

            if pricing_offered >= self.attrite_threshold:
                if self.exist == False:
                    self.exist = True
                    return +self.curr_amt,self.attrite_threshold,pricing_offered,self.curr_m
                else:
                    self.exist = True
                    return 0,self.attrite_threshold,pricing_offered,self.curr_m
            else:
                ##withdrawal and join other bank at the pricing ask
                if self.exist == True:
                    self.exist = False
                    return -self.curr_amt,self.attrite_threshold,pricing_offered,self.curr_m
                else:
                    self.exist = False
                    return 0,self.attrite_threshold,pricing_offered,self.curr_m
        return 0,self.attrite_threshold,0,self.curr_m

    #3B.iv generate the amount the bank has to pay to the depoistor as expense
    def generate_expense(self,curr_dte):
        expense = 0
        duration=0
        amt = 0
        if curr_dte <= self.m_dte and self.exist:
            expense = self.curr_amt * self.curr_pric / 100 /365
            duration = self.curr_amt * (self.m_dte - curr_dte).days/365
            amt = self.curr_amt
        return expense,duration,amt

##3C - borrower
#class borrower simulate for behavior of one borrower
class borrower(object):

    #3C.i init
    def __init__(self,curr_amt, curr_pric,curr_m, start_dte):
        self.curr_amt = curr_amt
        self.curr_pric = curr_pric
        self.curr_m = curr_m
        self.prob_attrite = random.uniform(0,1)
        self.start_dte = start_dte
        self.m_dte = self.start_dte + datetime.timedelta(days=self.curr_m)
        self.exist = False
        self.attrite_threshold = curr_pric
        return

    #3C.ii set loan expectation given the market pricing
    #response with the loan expectation, as well as amount and maturity
    #expectation will be updated only when it is maturing today
    def set_loan_expectation(self,curr_dte,mkt_loan_pricing_grid):
        if curr_dte == self.m_dte:
            #MAX_INT = 60
            mkt_pricing = mkt_loan_pricing_grid[self.curr_m]
            self.prob_attrite = random.uniform(0,1)
            prob_retention = 1 - self.prob_attrite
            pricing_considered = mkt_pricing* 1.05
            #assume that standard deviation of the pricing sensitivty is 1/4 of its currently given pricing
            if simulation_mc == True:
                attrite_threshold = pricing_sensitivity_mc(prob_retention, pricing_considered)
            else:
                attrite_threshold = pricing_sensitivity_linear(prob_retention, pricing_considered)
            self.attrite_threshold = attrite_threshold
            return self.attrite_threshold, self.curr_m, self.curr_amt
        else:
            #no expectation if not yet mature
            self.attrite_threshold = 60
            return -1,self.curr_m, self.curr_amt

    #3C.ii set loan expectation given the market pricing
    #response with the loan expectation, as well as amount and maturity
    #expectation will be updated only when it is maturing today
    def get_loan_maturing(self,tomorrow_dte):
        if tomorrow_dte == self.m_dte:
            return self.curr_m, self.curr_amt
        else:
            #no expectation if not yet mature
            return self.curr_m, 0

    #3C.iii make the roll over or not decision based on the expectation and pricing offered by the bank
    #run at maturity date to determine if there is any need for refinancing (financing the same loan after maturity)
    #expectation will be updated only when it is maturing today
    def set_loan_roll_over(self,pricing_dict, curr_dte):
        if self.m_dte <= curr_dte:
            pricing_offered = 0
            try:
                pricing_offered = pricing_dict[self.curr_m]
            except Exception:
                print(self.curr_m)

            self.start_dte = curr_dte
            self.curr_pric = self.attrite_threshold
            self.m_dte = self.start_dte + datetime.timedelta(days=self.curr_m)
            self.prob_attrite = random.uniform(0,1)

            if pricing_offered <= self.attrite_threshold:
                if self.exist == False:
                    self.exist = True
                    return +self.curr_amt,self.attrite_threshold,pricing_offered,self.curr_m
                else:
                    self.exist = True
                    return 0,self.attrite_threshold,pricing_offered,self.curr_m
            else:
                ##withdrawal and join other bank at the pricing ask
                if self.exist == True:
                    self.exist = False
                    return -self.curr_amt,self.attrite_threshold,pricing_offered,self.curr_m
                else:
                    self.exist = False
                    return 0,self.attrite_threshold,pricing_offered,self.curr_m
        return 0,self.attrite_threshold,0,self.curr_m

    #3C.iv generate the amount the bank has to lend to the borrower as income
    #run at any day before or at maturity day - to generate interest income
    def generate_income(self,curr_dte):
        income = 0
        duration=0
        amt = 0
        if curr_dte <= self.m_dte and self.exist:
            income = self.curr_amt * self.curr_pric / 100 /365
            duration = self.curr_amt * (self.m_dte - curr_dte).days/365
            amt = self.curr_amt
        return income,duration,amt

##3D - bank
#class bank simulate for behavior of bank or market (of banks)
'''
i. init
ii. pricing
iii. P&L
iv. output
'''
class bank(object):

    #3D.i init
    def __init__(self,dep_model_path=' ',loan_model_path=' '):
        self.income = 0.0
        self.expense = 0.0
        self.loan = 0.0
        self.deposit = 0.0
        
        self.prev_income = 0.0
        self.prev_expense = 0.0
        self.prev_loan = 0.0
        self.prev_deposit = 0.0
        
        self.loan_duration = 0.0
        self.deposit_duration = 0.0
        self.np_deposit_grid = []
        self.np_loan_grid = []
        
        self.self_funding_target = 0.0

        #create a new deposit model if no file path to an existing model is given
        if dep_model_path == ' ':
            inputs = Input(batch_shape=(1,36))
            x = Dense(10, activation='relu')(inputs)
            x = Dense(15, activation='relu')(x)
            predictions = Dense(units=15, activation='sigmoid')(x)
            dep_model= Model(inputs=inputs,outputs=predictions)
            dep_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            self.deposit_model = dep_model
        else:
            self.deposit_model = load_model(dep_model_path)

        #create a new loan model if no file path to an existing model is given
        if loan_model_path == ' ':
            inputs = Input(batch_shape=(1,36))
            x = Dense(10, activation='relu')(inputs)
            x = Dense(15, activation='relu')(x)
            predictions = Dense(units=15, activation='linear')(x)
            loan_model= Model(inputs=inputs,outputs=predictions)
            loan_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            self.loan_model = loan_model
        else:
            self.loan_model = load_model(loan_model_path)

        self.expense_chg=0.0
        self.prev_expense_chg=0.0
        self.deposit_chg=0.0
        self.prev_deposit_chg=0.0

        self.income_chg=0.0
        self.prev_income_chg=0.0
        self.loan_chg=0.0
        self.prev_loan_chg=0.0

        self.profitability = 0.0
        self.prev_profitability=0.0

        self.loan_grid_update =np.zeros(shape=(1,15))
        self.deposit_grid_update =np.zeros(shape=(1,15))

        self.loan_grid_update_negative =np.zeros(shape=(1,15))
        self.deposit_grid_update_negative =np.zeros(shape=(1,15))

        self.x_np_deposit =np.zeros(shape=(1,36))
        self.x_np_loan =np.zeros(shape=(1,36))

        self.x_np_deposit_negative =np.zeros(shape=(1,36))
        self.x_np_loan_negative =np.zeros(shape=(1,36))
        return

    ##3D.ii generate pricing grids
    ##3D.iia static pricing grids for loan and deposit 
    ##for static deposit pricing grid
    def generate_deposit_grid(self,pricing_grid):
        self.deposit_grid = pricing_grid
        self.deposit_grid_original = pricing_grid
        return self.deposit_grid

    ##3D.iia for static loan pricing grid
    def generate_loan_grid(self,pricing_grid):
        #15 pricing points
        self.loan_grid = pricing_grid
        self.loan_grid_original = pricing_grid
        return self.loan_grid

    ##3D.ii update market pricing for both market and bank
    ##3D.ii.b update market pricing for market: pricing grid update according to weiner process
    def generate_pricing_grids_MC(self):
        for key, value in self.deposit_grid.items():
            rnd = random.uniform(0,1)
            if simulation_mc==True:
                #self.deposit_grid[key] = pricing_sensitivity_mc(rnd,self.deposit_grid_original[key])
                self.deposit_grid[key] = pricing_sensitivity_mc(rnd,value)
            else:
                #self.deposit_grid[key] = pricing_sensitivity_linear(rnd,self.deposit_grid_original[key])
                self.deposit_grid[key] = pricing_sensitivity_linear(rnd,value)

        for key_l, value_l in self.loan_grid.items():
            rnd_l = random.uniform(0,1)
            if simulation_mc==True:
                #self.loan_grid[key] = pricing_sensitivity_mc(rnd_l,self.loan_grid_original[key])
                self.loan_grid[key] = pricing_sensitivity_mc(rnd_l,value_l)
            else:
                #self.loan_grid[key] = pricing_sensitivity_linear(rnd_l,self.loan_grid_original[key])
                self.loan_grid[key] = pricing_sensitivity_linear(rnd_l,value_l)
        return self.deposit_grid,self.loan_grid

    #3D.ii.c update market pricing for bank: use ml method to generate the grid
    # deposit: run prediction for deposit
    def generate_deposit_grid_ML(self,maturing_depositors_amt,mkt_grid,day_cnt,self_funding_target):
        x_np_deposit = [day_cnt, self.deposit,self.expense,self.loan,self.income,self_funding_target]
        for key, value in maturing_depositors_amt.items():
            x_np_deposit.append(value)
        for key, value in mkt_grid.items():
            x_np_deposit.append(value)
        x_np_deposit = np.array(x_np_deposit,dtype=float).reshape(1,36)
        self.x_np_deposit = x_np_deposit
        #output is a 2d matrix, need to specify that it has only one row
        self.deposit_grid_update = self.deposit_model.predict(x_np_deposit)[0]
        ##update_deposit_grid by shift
        i = 0
        for key, value in self.deposit_grid.items():
            self.deposit_grid[key] = self.deposit_grid_update[i]
            i+=1
        return self.deposit_grid,x_np_deposit

    #loan: run prediction for loan
    def generate_loan_grid_ML(self,maturing_borrowers_amt, mkt_grid,day_cnt,self_funding_target):
        x_np_loan = [day_cnt, self.deposit,self.expense,self.loan,self.income,self_funding_target]
        for key, value in maturing_borrowers_amt.items():
            x_np_loan.append(value)
        for key, value in mkt_grid.items():
            x_np_loan.append(value)

        x_np_loan = np.array(x_np_loan,dtype=float).reshape(1,36)
        self.x_np_loan= x_np_loan
        #output is a 2d matrix, need to specify that it has only one row
        self.loan_grid_update = self.loan_model.predict(x_np_loan)[0]
        ##update_loan_grid by shift
        i = 0
        for key_l, value_l in self.loan_grid.items():
            self.loan_grid[key_l] = self.loan_grid_update[i]
            i+=1
        return self.loan_grid,x_np_loan

    #3D.ii.d. Feedback / reinforcement
    #prep x as numpy for feedback
    def ML_prep_x_np(self,maturing_amt, mkt_grid,day_cnt,self_funding_target):
        x_np = [day_cnt, self.deposit,self.expense,self.loan,self.income,self_funding_target]
        for key, value in maturing_amt.items():
            x_np.append(value)
        for key, value in mkt_grid.items():
            x_np.append(value)
        x_np = np.array(x_np_loan,dtype=float).reshape(1,36)
        return x_np
        
    #prep y as numpy for feedback
    def ML_prep_y_np(self,pricing_grid):
        y_np = np.zeros((1,15), dtype=float)
        i=0
        for key, value in pricing_grid.items():
            y_np[0][i] = value
            i+=1
        y_np = np.array(y_np,dtype=float).reshape(1,15)
        return y_np


    #3D.ii.d. feedback: feed training set as reinforcement for both models
    def ML_feedback(self,maturing_borrowers_amt, maturing_depositors_amt,mk_loan_grid,mk_deposit_grid,loan_grid_final, deposit_grid_final, day_cnt, self_funding_target, epoch=1):
        #better trend, self-funding and profitable
        x_np_loan = self.ML_prep_x_np(maturing_borrowers_amt,mk_loan_grid,day_cnt, self_funding_target)
        y_np_loan = self.ML_prep_y_np(loan_grid_final)
        
        x_np_deposit = self.ML_prep_x_np(maturing_depositors_amt,mk_deposit_grid,day_cnt, self_funding_target)
        y_np_deposit = self.ML_prep_y_np(deposit_grid_final)
        
        self.deposit_model.fit(y=y_np_deposit, x=x_np_deposit , epochs=epoch,verbose=0)
        self.loan_model.fit(y=y_np_loan, x=x_np_loan, epochs=epoch,verbose=0)
        
        self.loan_grid = loan_grid_final
        self.deposit_grid = deposit_grid_final
        return True

    ##3D ii.e.Generate pricing grids based on existing grid
    def MC_pricing_grid_variations(self,temp_deposit_grid,temp_loan_grid):
        for key, value in temp_deposit_grid.items():
            rnd = random.uniform(0,1)
            if simulation_mc==True:
                temp_deposit_grid[key] = pricing_sensitivity_mc(rnd,self.deposit_grid[key])
            else:
                temp_deposit_grid[key] = pricing_sensitivity_linear(rnd,self.deposit_grid[key])
            

        for key_l, value_l in temp_loan_grid.items():
            rnd_l = random.uniform(0,1)
            if simulation_mc == True:
                temp_loan_grid[key] = pricing_sensitivity_mc(rnd_l,self.loan_grid[key])
            else:
                temp_loan_grid[key_l] = pricing_sensitivity_linear(rnd_l,self.loan_grid[key])
        return temp_deposit_grid,temp_loan_grid

    ##3D.iii set deposit or loan P&L
    def set_deposit_PnL(self,deposit,expense):
        self.prev_deposit_chg = self.deposit_chg
        self.prev_expense_chg = self.expense_chg
        self.deposit_chg = deposit
        self.expense_chg = expense
        self.deposit = self.deposit + deposit
        self.expense = self.expense + expense
        return self.deposit, self.expense

    def set_loan_PnL(self,loan,income):
        self.prev_loan_chg = self.loan_chg
        self.prev_income_chg = self.income_chg
        self.loan_chg = loan
        self.income_chg = income
        self.loan = self.loan + self.loan_chg
        self.income = self.income + self.income_chg
        return self.loan, self.income

    ##3D.iv. Output functions
    #Output grides
    def output_grids(self):
        output_str=""
        for i in self.deposit_grid:
            output_str = output_str + str(i) + ">>"+str(self.deposit_grid[i]) + '|'
        output_str =output_str + '********'
        for i in self.loan_grid:
            output_str = output_str + str(i) + ">>"+str(self.loan_grid[i]) + '|'
        return output_str

    #Save models
    def save_models(self,dep_path,loan_path):
        self.deposit_model.save(dep_path)
        self.loan_model.save(loan_path)
        return True

##3E: Environment Objecct will provide the reward estimates, given the environment variables, including the pricing grids of loan and deposit
    #it will generate two output: 1. the Net Profit of this loan and pricing grids, as well as the self funding ratio
class environment(object):
    def __init__(self,reward_model_input_features,reward_model_path ='',fund_model_path=''):
        #create a new deposit model if no file path to an existing model is given
        if reward_model_path == '':
            #profiability model
            inputs = Input(batch_shape=(1,reward_model_input_features))
            x = Dense(15, activation='relu')(inputs)
            x = Dense(15, activation='relu')(x)
            x = Dense(15, activation='relu')(x)
            predictions = Dense(units=1, activation='linear')(x)
            reward_model= Model(inputs=inputs,outputs=predictions)
            reward_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            self.reward_model = reward_model

            #funding model
            inputs = Input(batch_shape=(1,reward_model_input_features))
            x = Dense(15, activation='relu')(inputs)
            x = Dense(15, activation='relu')(x)
            x = Dense(15, activation='relu')(x)
            predictions = Dense(units=1, activation='linear')(x)
            funding_model= Model(inputs=inputs,outputs=predictions)
            funding_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
            self.funding_model = funding_model
        else:
            #load them otherwise
            self.reward_model = load_model(reward_model_path)
            self.funding_model = load_model(funding_model_path)
        self.reward_model_input_features = reward_model_input_features
        return
    
    #environ_list = 
    #day_cnt, deposit, expense, loan, income, self_funding_target, net_PnL (previous)
    #grids:maturing_borrowers_amt, maturing_depositors_amt, mkt_loan_grid, mkt_deposit_grid, bk_loan_grid, bk_deposit_grid
    def generate_rewards_ML(self,environ_list, environ_grid_list):
        x_np = environ_list
        for grid in environ_grid_list:
            for key, value in grid.items():
                x_np.append(value)
        x_np = np.array(x_np,dtype=float).reshape(1,self.reward_model_input_features)
        self.x_np= x_np

        #output is a 2d matrix, need to specify that it has only one row
        self.y_pred = self.reward_model.predict(x_np)[0]
        PnL_pred = self.y_pred
        self.y_pred = self.funding_model.predict(x_np)[0]
        self_funding_pred = self.y_pred
        return x_np, PnL_pred, self_funding_pred

    #3D.iv.b feedback: feed training set as reinforcement for both models
    def ML_reward_feedback(self,x_np,PnL_actual, self_funding_actual,epoch=1):
        
        y_actual = np.zeros((1,1),dtype=float)
        y_actual[0,0]=PnL_actual
        y_actual = y_actual.reshape(1,1)
        x_np = np.asarray(self.x_np).reshape(1,self.reward_model_input_features) 
        self.reward_model.fit(y=y_actual, x=x_np , epochs=epoch,verbose=0)

        y_actual = np.zeros((1,1),dtype=float)
        y_actual[0,0]=self_funding_actual
        y_actual = y_actual.reshape(1,1)
        x_np = np.asarray(self.x_np).reshape(1,self.reward_model_input_features)    
        self.funding_model.fit(y=y_actual, x=x_np , epochs=epoch,verbose=0)
        return True

    #3D.v.b: calculate the amount of deposit up for repricing
    def save_models(self,reward_path,funding_path):
        self.reward_model.save(reward_path)
        self.reward_model.save(funding_path)
        return True

##3F - load the list of loans/depositors as list of objects
#function to generate list of depositors and list of borrowers from dataframe
def generate_list(f_deposit_path,f_loan_path,start_date):
    list_depositors = []
    df_deposit = pd.read_csv(f_deposit_path,sep=',')
    df_deposit['Start_Date']=pd.to_datetime(df_deposit['Start_Date'],format='%m/%d/%y',errors='coerce')
    for index, row in df_deposit.iterrows():
        d_curr_amt =row['Deposit_Amt']
        d_curr_m = row['Duration_Day']
        d_curr_pric = row['Pricing']
        d_start_dte = start_date
        tmp_depositor = depositor(d_curr_amt, d_curr_pric,d_curr_m, d_start_dte)
        list_depositors.append(tmp_depositor)

    list_borrowers = []
    df_loan = pd.read_csv(f_loan_path,sep=',')
    df_loan['Start_Date']=pd.to_datetime(df_loan['Start_Date'],format='%m/%d/%y',errors='coerce')
    df_loan['Duration_Day'] =df_loan['Duration_Day'].astype(int)
    for index, row in df_loan.iterrows():
        l_curr_amt =row['Loan_Amt']
        l_curr_m = row['Duration_Day']
        l_curr_pric = row['Pricing']
        l_start_dte = start_date
        tmp_borrower = borrower(l_curr_amt, l_curr_pric,l_curr_m, l_start_dte)
        list_borrowers.append(tmp_borrower)
    return list_depositors,list_borrowers


'''*************************************
##Section 4. Run the functions above
'''
'''*************************************
##Step 1: Import Data from the list of Loans and Deposits
##Based on an input file, generate list of borrowers and depositors at the beginning
##Keep the master copy of clean clients list
'''
list_depositors_template,list_borrowers_template = generate_list(f_deposit_path,f_loan_path,start_date)

'''*************************************
##Step 2:. Run 1000 simulations
'''
print('running simulation')
for run in range(0,1000):
    print('simulation ' +str(run))

    #reward function reset
    reward = 0
    list_depositors = copy.deepcopy(list_depositors_template)
    list_borrowers = copy.deepcopy(list_borrowers_template)
    deposit, expense,loan,income = 0,0,0,0
    cum_PnL = 0
    prev_cum_PnL =0
    ##Step 3: Build the model
    #build a model if this is the first run, otherwise, load the saved model
    #bank and environment objects created
    if run==0:
        jpm = bank()
        env = environment(reward_model_input_features)
    else:
        jpm = bank(dep_model_path, loan_model_path)
        #env = environment(reward_model_input_features,reward_model_path)
        prev_failed_bank = False

    deposit_pricing_grid_pred = jpm.generate_deposit_grid(deposit_constant_grid)
    loan_pricing_grid_pred = jpm.generate_loan_grid(loan_constant_grid)
    loan_pricing_grid_prev = loan_empty_grid
    deposit_pricing_grid_prev = deposit_empty_grid
    loan_pricing_grid_final = loan_empty_grid
    deposit_pricing_grid_final = deposit_empty_grid

    #market is also a bank (collective behavior of many banks)
    #market object created
    market = bank()
    mkt_deposit_pricing_grid = market.generate_deposit_grid(deposit_constant_grid)
    mkt_loan_pricing_grid = market.generate_loan_grid(loan_constant_grid)

    daily_loan_list=[]
    daily_deposit_list=[]
    daily_net_asset_list=[]
    cum_income_earned =0
    cum_expense_paid =0

    mkt_expense = 0
    mkt_income = 0

    for i_depositor in list_depositors_template:
        today_m, today_amt = i_depositor.get_deposit_maturing(start_date)
        maturing_depositors_amt[today_m] += today_amt

    for i_borrower in list_borrowers_template:
        today_m, today_amt = i_borrower.get_loan_maturing(start_date)
        maturing_borrowers_amt[today_m] += today_amt

    loan_grid_attric = loan_empty_grid
    deposit_grid_attric = deposit_empty_grid
    
    #simulate for 10 years
    for day_cnt in range(0,3650):
        print(day_cnt)
        today_dte = start_date + datetime.timedelta(days=day_cnt)
        loan_grid_choice=""
        deposit_grid_choice=""

        tomorrow_dte = today_dte + datetime.timedelta(days=1)

        deposit_chg = 0
        deposit_duration = 0
        loan_duration = 0
        loan_chg = 0
        daily_loan_chg = 0
        daily_deposit_chg = 0
        income_earned = 0
        expense_paid = 0
        loan_os = 0
        deposit_os = 0
        
        daily_income_positive = 0
        daily_expense_positive = 0
        daily_loan_positive = 0
        daily_deposit_positive = 0

        daily_income_negative = 0
        daily_expense_negative = 0
        daily_loan_negative = 0
        daily_deposit_negative = 0
        
        #self funding ratio - ratio of deposit to loan, inrease by 1% every day
        self_funding_target = min(self_funding_target + day_cnt*0.01,1)
      
        ''''*************************************
        ##Step 3: generate the pricing grids for the day
        '''
        ##3A. Generate the pricing grids by bank and market
        ##Generate two pricing grids for the day
        mkt_deposit_pricing_grid, mkt_loan_pricing_grid = market.generate_pricing_grids_MC()
        loan_pricing_grid_pred,x_np_loan = jpm.generate_loan_grid_ML(maturing_borrowers_amt,mkt_loan_pricing_grid,day_cnt,self_funding_target)
        deposit_pricing_grid_pred,x_np_deposit = jpm.generate_deposit_grid_ML(maturing_depositors_amt,mkt_deposit_pricing_grid,day_cnt,self_funding_target)
        loan_pricing_grid_prev = loan_pricing_grid_final
        deposit_pricing_grid_prev = deposit_pricing_grid_final

        #maturiing deposit amount by maturity date
        today_maturing_borrowers_amt = maturing_borrowers_amt
        today_maturing_depositors_amt = maturing_depositors_amt 
        maturing_borrowers_amt = dict(loan_empty_grid)
        maturing_depositors_amt = dict(deposit_empty_grid)
        maturing_borrowers_pric = dict(loan_empty_grid)
        maturing_depositors_pric = dict(deposit_empty_grid)
        
        ##3B. Estimate the maturity profile of the clients
        maturing_borrower_cnt = 0
        for i_borrower in list_borrowers:
            
            #set pricing expectation
            exp_pric, exp_m, exp_amt = i_borrower.set_loan_expectation(today_dte,mkt_loan_pricing_grid)

            #add list of borrower up maturing tomorrow - as features to fed to the pricing engine to consider
            #exp_pric >-1 if maturing today
            if exp_pric > -1:
                maturing_borrowers_amt[exp_m] += exp_amt
                maturing_borrower_cnt+=1
                maturing_borrowers_pric[exp_m] += exp_pric * exp_amt

        #calculate the weighted average pricing for borrowers
        for key, value in maturing_borrowers_pric.items():
            if maturing_borrowers_amt[key] > 0:
                maturing_borrowers_pric[key]=value/maturing_borrowers_amt[key]
                
        maturing_depositor_cnt = 0    
        for i_depositor in list_depositors:
                        
            #set pricing expectation
            exp_pric, exp_m, exp_amt = i_depositor.set_deposit_expectation(today_dte,mkt_deposit_pricing_grid)
            
            #add list of borrower up maturing tomorrow - as features to fed to the pricing engine to consider
            #exp_pric >-1 if maturing today
            if exp_pric > -1:
                maturing_depositors_amt[exp_m] += exp_amt
                maturing_depositor_cnt+=1
                maturing_depositors_pric[exp_m] += exp_pric * exp_amt
        
        #calculate the weighted average pricing for depositors
        for key, value in maturing_depositors_pric.items():
            if maturing_depositors_amt[key] > 0:
                maturing_depositors_pric[key]=maturing_depositors_pric[key]/maturing_depositors_amt[key]
        
        '''*************************************
        ##Step 4:. Generate list of possible pricing, predict the reward and pick the best pricing (action)
        '''        
        ##4A. Generating list of all possible loan / deposit pricing, including previous, and current predicted pricing
        loan_pricing_grid_list = [loan_pricing_grid_prev,loan_pricing_grid_pred]
        deposit_pricing_grid_list = [deposit_pricing_grid_prev,deposit_pricing_grid_pred]
        #generate lots of variations:
        for i in range(0,num_randomized_grid):
            temp_loan_grid,temp_deposit_grid = jpm.MC_pricing_grid_variations(loan_empty_grid, deposit_empty_grid)
            loan_pricing_grid_list.append(temp_loan_grid)
            deposit_pricing_grid_list.append(temp_deposit_grid)
        
        #accessing each of the choice
        result_dict={}
        x_np_reward_final = np.zeros((1,1))
        x_np_reward_list = []

        loan_i_final = 0
        deposit_i_final = 0
        max_reward=0
        c1=0
        c2=0

        num_grid_variations = len(loan_pricing_grid_list)
        
        ##4B. Predict the reward values of each the variation and make the choice of pricing
        for loan_i in range(0,num_grid_variations):
            for deposit_i in range(0,num_grid_variations):
                temp_reward=0
                bk_loan_grid = loan_pricing_grid_list[loan_i]
                bk_deposit_grid = deposit_pricing_grid_list[deposit_i]
                reward_environ_list = [day_cnt, deposit, expense, loan,income, self_funding_target, prev_cum_PnL]
                reward_environ_grid_list = [maturing_borrowers_amt, maturing_depositors_amt,maturing_borrowers_pric, maturing_depositors_pric, mkt_loan_pricing_grid, mkt_deposit_pricing_grid, bk_loan_grid,bk_deposit_grid]                
                temp_x_np, temp_cum_PnL, temp_self_funding_target = env.generate_rewards_ML(reward_environ_list, reward_environ_grid_list)
                x_np_reward_list.append(temp_x_np)
             
                if temp_cum_PnL > 0:
                    c1 =1
                if temp_self_funding_target > self_funding_target:
                    c2=1
                temp_reward = c1 * c2 * temp_cum_PnL
                #Policy A
                if max_reward<= temp_reward:
                    loan_i_final = loan_i
                    deposit_i_final = deposit_i
                    max_reward = temp_reward
                    #funding_target_final = temp_self_funding_target
                    #x_np_final = temp_x_np
                    
        #Policy B: if both conditions fail, randomize the choice
        if max_reward == 0:
            tmp_loan_seed = random.uniform(0,num_grid_variations)
            tmp_deposit_seed = random.uniform(0,num_grid_variations)
            loan_i_final = int(tmp_loan_seed)
            deposit_i_final = int(tmp_deposit_seed)
            
        ##4C. Policy: Choose the best choice & reward
        loan_pricing_grid_final = loan_pricing_grid_list[loan_i_final]
        deposit_pricing_grid_final = deposit_pricing_grid_list[deposit_i_final]
        #print(len(loan_pricing_grid_final))
        reward_environ_list = [day_cnt, deposit, expense, loan,income, self_funding_target, prev_cum_PnL]
        reward_environ_grid_list = [maturing_borrowers_amt, maturing_depositors_amt,maturing_borrowers_pric, maturing_depositors_pric, mkt_loan_pricing_grid, mkt_deposit_pricing_grid, loan_pricing_grid_final,deposit_pricing_grid_final]
        x_np_final, pred_cum_PnL, pred_self_funding_target = env.generate_rewards_ML(reward_environ_list, reward_environ_grid_list)
        debug1 = len(reward_environ_grid_list)
        
        '''****************************************
        Step 5. Execute the pricing grid
        '''
        #5A. Carry forward the deposit and Roll-over the loan
        #stay or not
        ##Update borrower and depositor
        for i_borrower in list_borrowers:
            loan_chg,attric_pric,pric_offer,m = i_borrower.set_loan_roll_over(loan_pricing_grid_final,today_dte)
            daily_loan_chg += loan_chg
            if attric_pric < loan_grid_attric[m] and loan_grid_attric[m] != 0:
                loan_grid_attric[m] = attric_pric
            
        for i_depositor in list_depositors:
            deposit_chg,attric_pric,pric_offer,m = i_depositor.set_deposit_carry_fwd(deposit_pricing_grid_final,today_dte)
            daily_deposit_chg += deposit_chg
            if attric_pric > deposit_grid_attric[m] and deposit_grid_attric[m] != 0:
                deposit_grid_attric[m] = attric_pric
        
        #5B. Actualized p n l
        ##*************************************
        #5Bi. with clients
        for i_borrower in list_borrowers:
            #pocket in the loan interest
            i_income,i_duration,i_loan = i_borrower.generate_income(today_dte)
            loan_duration += i_duration
            income_earned += i_income
            loan_os += i_loan
 
        for i_depositor in list_depositors:
            #pay out the deposit interest
            i_expense, i_duration,i_deposit = i_depositor.generate_expense(today_dte)
            deposit_duration += i_duration
            expense_paid += i_expense
            deposit_os += i_deposit
        
        #6Bii. with market
        #market operations
        net_loan = loan_os - deposit_os
        if net_loan > 0:
            mkt_deposit = net_loan
            mkt_loan = 0
        else:
            mkt_deposit = 0
            mkt_loan = net_loan*-1
        #overnight funding
        mkt_expense = mkt_loan * mkt_deposit_pricing_grid[0]/100/365
        #overnight lending
        mkt_income = mkt_deposit * mkt_loan_pricing_grid[0]/100/365
        
        ##*************************************
        #5C: End of day closing
        ##*************************************
        #cumulative income = income earned from client + income earned from market (if any excess deposit placed overnight)
        daily_total_income = income_earned+mkt_income
        cum_income_earned+= daily_total_income

        #cumulative expense = expense paid to the client + expense paid to market (if any insufficient deposit to fund overnight pos)
        daily_total_expense = expense_paid + mkt_expense
        cum_expense_paid += daily_total_expense
        
        prev_cum_PnL = cum_PnL
        cum_PnL = cum_income_earned - cum_expense_paid
        daily_total_PnL = daily_total_income - daily_total_expense
        
        #Closed book for the day
        loan, income = jpm.set_loan_PnL(daily_loan_chg,income_earned+mkt_income)
        deposit, expense = jpm.set_deposit_PnL(daily_deposit_chg,expense_paid+mkt_expense)

        if loan_os == 0:
            loan_os = 0.0000000000000000000000000001

        f_log.write('\n****************summary run:' +str(run) + ' day ' +str(day_cnt)+'****************')
        f_log.write('\tpred_cum_PnL:'+str(pred_cum_PnL))
        f_log.write('\tactual_cum_PnL:'+str(cum_PnL))
        f_log.write('\tpred_self_funding_target:'+str(pred_self_funding_target))
        f_log.write('\tactual_cum_PnL:'+str(deposit_os/loan_os))
        f_log.write('\tloan os:'+str(loan_os))
        f_log.write('\tdeposit os:' + str(deposit_os))
        f_log.write('\tcum income earned '+str(cum_income_earned))
        f_log.write('\tincome earned:'+str(income_earned))
        f_log.write("\tmkt loan pricing:" + str(mkt_loan_pricing_grid))
        f_log.write("\tloan pricing:" + str(loan_pricing_grid_final))
        f_log.write('\tcum expense paid:' + str(cum_expense_paid))
        f_log.write('\texpense paid:' + str(expense_paid))
        f_log.write("\tmkt dep pricing:" +str(mkt_deposit_pricing_grid))
        f_log.write("\tdep pricing:" + str(deposit_pricing_grid_final))
        
        deposit_pricing_grid_prev = deposit_pricing_grid_final
        loan_pricing_grid_prev = loan_pricing_grid_final
        
        ##*************************************
        ##Step 6: Feedback/ Reinforcement
        ##*************************************
        #at the end of the day, if it passes the basic conditions (the bank survive), update parameters of the bank and customers
        #maturing_borrowers_amt, maturing_depositors_amt,mk_loan_grid,mk_deposit_grid,loan_grid_final, deposit_grid_final, day_cnt, self_funding_target            
        jpm.ML_feedback(today_maturing_borrowers_amt,today_maturing_depositors_amt,mkt_loan_pricing_grid, mkt_deposit_pricing_grid,loan_pricing_grid_final,deposit_pricing_grid_final, self_funding_target,1)

        actual_self_funding = deposit_os/loan_os
        env.ML_reward_feedback(x_np_final, daily_total_PnL, actual_self_funding)

        #failed the bank (break this simulation) if it cannot meet self-funding requirements
        if actual_self_funding < self_funding_target:
            #print('Failed bank at day ' + str(day_cnt))
            prev_failed_bank = True
        else:
            prev_failed_bank = False
            
    ##*************************************
    #Step 7: Output the result for the simulation
    #end of the run
    #output result of this run and save model
    print('run ' + str(run) + ' is completed')
    f_output.write(str(run))
    f_output.write('\t')
    f_output.write(str(day_cnt))
    f_output.write('\t')
    f_output.write(str(cum_income_earned-cum_expense_paid))
    f_output.write('\t')
    f_output.write(str(deposit_os/loan_os))
    f_output.write('\t')
    f_output.write(str(deposit_os))
    f_output.write('\t')
    f_output.write(str(cum_expense_paid))
    f_output.write('\t')
    f_output.write(str(deposit_duration))
    f_output.write('\t')
    f_output.write(str(loan_os))
    f_output.write('\t')
    f_output.write(str(cum_income_earned))
    f_output.write('\t')
    f_output.write(str(loan_duration))
    f_output.write('\t')
    f_output.write(str(mkt_loan)+">>"+str(mkt_deposit))
    f_output.write('\t')
    f_output.write(jpm.output_grids())
    f_output.write('\t')
    f_output.write(market.output_grids())
    f_output.write('\t')
    for k, v in maturing_depositors_amt.items():
        f_output.write(str(k) + ' >>> '+ str(v) + '|')
    f_output.write('\t')
    for k, v in maturing_borrowers_amt.items():
        f_output.write(str(k) + ' >>> '+ str(v) + '|')
    f_output.write('\n')
    jpm.save_models(dep_model_path, loan_model_path)
    env.save_models(reward_model_path,funding_model_path)
    del jpm
f_output.close()
f_log.close()
