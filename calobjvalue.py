# -*- coding: utf-8 -*-
"""
@author: Range_Young
"""

import numpy as np
import math

def sigmf(x):
    #return ((math.exp(x)-math.exp(-x)) / (math.exp(x)+math.exp(-x)))
    return 1 / (1+math.exp(-5*x))
def calobjvalue(Pop,  Data1, Data2, NumTime, AgentSize, iNode, NumNode, Response):
    obj = np.zeros([AgentSize, AgentSize])
    idTimeData = Data1[:, iNode].copy()
    # error is used to record the difference between TimeData and idTimeData of ith Node
    error = np.zeros(NumTime)
    sumE = np.zeros(Response)
    # calculate the error for each agent
    for i in xrange(AgentSize):
        for j in xrange(AgentSize):
            TempNet = Pop[i][j][:].copy()
            for r in xrange(Response):
                start = r*NumNode
                end = start + NumNode
                #idTimeData[0] = Data1[0,start]
                for ii in xrange(NumTime-1):
                    idTimeData[ii] = sigmf(np.dot(Data1[ii, start:end], TempNet))
                    # calculate the fitness for each individual
                error = np.subtract(idTimeData, Data2[:, start + iNode])
                sumE[r] = np.vdot(error, error)
            obj[i][j] = np.sum(sumE)/Response
    return obj