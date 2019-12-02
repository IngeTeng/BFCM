# -*- coding: utf-8 -*-
"""
@author: Range_Young
"""

import numpy as np
from Selection import Nbor

def Crossover(PopChild, AgentSize, NumNode, ProCrossover, PopFit):
    for i in xrange(AgentSize):
        for j in xrange(AgentSize):
            mpos = np.random.randint(0, NumNode)
            if(np.random.rand() < ProCrossover):
                [xLabel, yLabel] = Nbor(i, j, AgentSize, PopFit)
                #AgentNeb = PopChild[xLabel, yLabel, :].copy()
                #PartVec = np.multiply(np.subtract(AgentNeb, PopChild[i, j, :]), np.random.rand(NumNode))
                #PopChild[i, j, :] = np.add(PopChild[i, j, :], PartVec)
                PopChild[i, j, 0:mpos] = PopChild[xLabel, yLabel, 0:mpos].copy()
    return PopChild