# -*- coding: utf-8 -*-
"""
@author: Range_Young
"""

if __name__ == "__main__":
    print("this is the main program")

import numpy as np
import scipy.io as sio
import copy as cp
import math

from calobjvalue import calobjvalue
from Mutate import Mutate
from Crossover import Crossover
from Selection import Selection

# sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-5*x))

#parameters
NumNetwork = 10
NumNode = 10
Response = 1
NumTime = 20

MaxGeneration = 500
AgentSize = 10
ProCrossover = 0.7
ProMutation = 0.6
#load data
loss = []
for st in range(20):
    name1 = 'FCMData'+str(st+1)+'.mat'
    name2 = 'LearnedNetwork'+str(st+1)+'.mat'
    name3 = 'aall'+str(st+1)+'.mat'
    TD = sio.loadmat(name1)
    TimeData = TD['a']
    TimeData = TimeData.T
    #initiation
    BestAgent = np.zeros(NumNode)
    BestFit = 0
    BestAgentFinal = BestAgent.copy()
    BestFitFinal = 0
    
    LearnedNetwork = np.zeros([NumNode, NumNode, NumNetwork])
    ww = np.zeros([NumNode, NumNetwork+1])
    ww[:,0] = 1
    
    aall = []
    
    for iNode in xrange(NumNode):
        # initialize the Population
        Data1 = TimeData[0:(NumTime-1),:].copy()
        Data2 = TimeData[1:NumTime,:].copy()
        training = Data2.copy()
        temp = np.zeros([NumTime-1,Response,NumNetwork])
        #sumer_all = []
        collect = []
        for iNet in range(NumNetwork):
            Pop = np.random.uniform(-1, 1, size=(AgentSize, AgentSize, NumNode))
            # PopChild
            PopChild = Pop.copy()
            # Fitness of Population
            PopFit = np.zeros([AgentSize, AgentSize])
            
            for Itr in xrange(MaxGeneration):
                print('Node:%d __ Itr:%d'%(iNode,Itr))
                # split the time  series data #
                # input into the following function and cal fit for each response
                # mean the PopFit
                # PopFit is an agentSize*AgentSize matrix
                PopFit = calobjvalue(Pop, Data1, training, NumTime, AgentSize, iNode, NumNode, Response)
                # find the best agent & its fitness
                BestFit = np.min(PopFit)
                # (x,y) is the pos of the best Agent
                x = np.argmin(PopFit) / AgentSize
                y = np.argmin(PopFit) % AgentSize
                BestAgent = Pop[x, y, :].copy()
                if (Itr==0):
                    BestAgentFinal = cp.deepcopy(BestAgent)
                    BestFitFinal = cp.deepcopy(BestFit)
                # self-learning on the best agent AND update BestAgent
                # [BestAgent, BestFit] = selflearning(BestAgent, BestFit, TimeData, NumNode, NumTime, iNode)
                # update BestAgentFinal
                print(str(BestFitFinal)+"—————"+str(BestFit))
                if BestFitFinal > BestFit:
                    BestFitFinal = cp.deepcopy(BestFit)
                    BestAgentFinal = cp.deepcopy(BestAgent)
                # select a neighbour for each individual in the pop
                # Pop->PopChild by selection
                PopChild = Selection(Pop, PopChild, PopFit, AgentSize)
                # crossover for PopChild
                PopChild = Crossover(PopChild, AgentSize, NumNode, ProCrossover, PopFit)
                # mutation for PopChild
                PopChild = Mutate(PopChild, AgentSize, NumNode, ProMutation, Itr)
                # PopChild->Pop
                Pop = PopChild.copy()
            LearnedNetwork[:, iNode, iNet] = BestAgentFinal.copy()
            TempNet = BestAgentFinal.copy()
            sumer = 0
            for r in xrange(Response):
                start = r*NumNode
                end = start + NumNode
                #idTimeData[0] = Data1[0,start]
                for ii in xrange(NumTime-1):
                    temp[ii,r,iNet] = sigmoid(np.dot(Data1[ii, start:end], TempNet))
                    tem = Data2[ii,(start+iNode)]-np.vdot(ww[iNode][0:NumNetwork],temp[ii,r,:])
                    training[ii,(start+iNode)] = tem
                    sumer = tem + sumer
            sumer = sumer/(NumTime*Response)
            #sumer_all.append(sumer)
        
            if sumer>0:
                ww[iNode, iNet+1]=np.e**(-sumer)
            else:
                if sumer<0:
                    ww[iNode, iNet+1]=-np.e**(sumer)
    
            for r in xrange(Response):
                start = r*NumNode
                end = start + NumNode
                #idTimeData[0] = Data1[0,start]
                for ii in xrange(NumTime-1):
                    if sumer>0 and (training[ii,(start+iNode)]<0):
                        training[ii,(start+iNode)] = 0
                    if sumer<0:
                        if training[ii,(start+iNode)]<0:
                            training[ii,(start+iNode)] = -training[ii,(start+iNode)]
                        else:
                            training[ii,(start+iNode)] = 0
            tmp = []
            Data3 = np.zeros([NumTime, Response])
            for r in range(Response):
                Data3[0,r] = TimeData[0,NumNode*r+iNode].copy()
            
            for r in xrange(Response):
                for iiNet in range(iNet+1):
                    start = r*NumNode
                    end = start + NumNode
                    for ii in xrange(NumTime-1):
                        Data3[(ii+1),r] = Data3[(ii+1),r]+ww[iNode,iiNet]*sigmoid(np.dot(Data1[ii, start:end],  LearnedNetwork[:, iNode, iiNet]))
                der = np.subtract(Data3[:,r],TimeData[:,(start+iNode)])
                der = np.vdot(der,der)
                tmp.append(der)
            collect.append(sum(tmp)/Response)
        aall.append(collect)
    
    
    LearnedTimeData=np.zeros([NumTime,NumNode*Response])
    LearnedTimeData[0,:] = TimeData[0,:].copy()
    for i in xrange(NumTime-1):
        for r in xrange(Response):
            start = r*NumNode
            end = start + NumNode
            tempData = LearnedTimeData[i, start:end].copy()
            for j in xrange(NumNode):
                for iNet in range(NumNetwork):
                    LearnedTimeData[i+1, start+j] = LearnedTimeData[i+1, start+j]+ ww[j][iNet]*sigmoid(np.dot(tempData, LearnedNetwork[:, j,iNet]))
                if LearnedTimeData[i+1, start+j]>1:
                    LearnedTimeData[i+1, start+j]=1
                if LearnedTimeData[i+1, start+j]<0:
                    LearnedTimeData[i+1, start+j]=0
    # print "Time series Data\n", LearnedTimeData
    # calculate the error
    der = np.subtract(TimeData[0:NumTime,:], LearnedTimeData)
    der = np.multiply(der, der)
    DataError = np.sum(der) * 1.0 / (NumNode*NumTime*Response)
    print("Data Error is "), DataError
    loss.append(DataError)
    sio.savemat(name2,{'net':LearnedNetwork})
    aall = np.array(aall)
    sio.savemat(name3,{'aall':aall})
loss=np.array(loss)
sio.savemat('loss.mat',{'loss':loss})
'''
select = [100*i for i in range(Responses)]
a = LearnedTimeData[:,select]
b = TimeData[:,select]

ans = abs(a-b)
print(sum(sum(ans)))
# ************************** diff of network structure ***************** #
# calculate the sensitivity specificity & ss mean
#sio.savemat('LearnedNetwork.mat',{'a':LearnedNetwork})


TP = 0
FN = 0
FP = 0
TN = 0
for i in xrange(NumNode):
    for j in xrange(NumNode):
        if np.fabs(LearnedNetwork[i, j]) > 0.05:
            LearnedNetwork[i, j] = 1
        else:
            LearnedNetwork[i, j] = 0
        # calculate spe, sen, ss mean
        if (LearnedNetwork[i, j] == 0) and (Network[i, j] != 0):
            FP = FP + 1
        elif (LearnedNetwork[i, j] == 0) and (Network[i, j] == 0):
            TP = TP + 1
        elif (LearnedNetwork[i, j] != 0) and (Network[i, j] != 0):
            TN = TN + 1
        else:
            FN = FN + 1
sen = TP*1.0 / (TP + FN)
spe = TN*1.0 / (TN + FP)
ss = (2*sen*spe*1.0) / (spe + sen)

print "the Learned Network\n", LearnedNetwork

print "FP, TP, TN, FN", FP, TP, TN, FN
print "sen is, spe, SS mean is, ", sen, spe, ss
'''
