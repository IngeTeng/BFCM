# -*- coding: utf-8 -*-
"""
@author: Range_Young
"""

import numpy as np

def Mutate(PopChild, AgentSize, NumNode, ProMutation, Itr):
    ps = int(NumNode*0.2)
    for i in xrange(AgentSize):
        for j in xrange(AgentSize):
            for ii in xrange(ps):
                mpoint = np.random.randint(0, NumNode)
                if(np.random.rand() < ProMutation):
                    PopChild[i, j, mpoint] = PopChild[i, j, mpoint] + np.random.uniform(-0.4, 0.4)
                    if np.fabs(PopChild[i, j, mpoint]) >= 1:
                        PopChild[i, j, mpoint] = np.random.uniform(-1, 1)
                if(np.random.rand() < 0.5):
                    PopChild[i, j, mpoint] = 0
    return PopChild