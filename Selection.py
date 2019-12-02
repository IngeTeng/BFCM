# -*- coding: utf-8 -*-
"""
@author: Range_Young
"""

import numpy as np

def Nbor(i, j, AgentSize, PopFit):
    meanFit = np.mean(PopFit)
    # ********* mean is changed to max ***** #
    # meanFit = np.max(PopFit)
    if (3>4):
        ix = np.random.randint(1, AgentSize)
        iy = np.random.randint(1, AgentSize)
    else:
        #list of Nbor
        one= np.random.randint(0, AgentSize)
        two= np.random.randint(0, AgentSize)
        three= np.random.randint(0, AgentSize)
        four= np.random.randint(0, AgentSize)
        one1= np.random.randint(0, AgentSize)
        two1= np.random.randint(0, AgentSize)
        three1= np.random.randint(0, AgentSize)
        four1= np.random.randint(0, AgentSize)
        NborList = np.array([[one, one1], [two, two1], [three, three1], [four, four1]])
        NborListFit = np.array([PopFit[one, one1], PopFit[two, two1], PopFit[three, three1], PopFit[four, four1]])
        if PopFit[i, j] <= meanFit:
            # select the individual with mini fitness
            minFit = np.argmin(NborListFit)
            if minFit == 0:
                ix = one
                iy = one1
            elif minFit == 1:
                ix = two
                iy = two1
            elif minFit == 2:
                ix = three
                iy = three1
            else:
                ix = four
                iy = four1
        else:
            r = np.random.randint(0, 4)
            ix = NborList[r][0]
            iy = NborList[r][1]

    return [ix, iy]

# selection process
def Selection(Pop, PopChild, PopFit, AgentSize):
    for i in xrange(AgentSize):
        for j in xrange(AgentSize):
            # decide how many neighbours Pop[i, j] should have
            [ix, iy] = Nbor(i, j, AgentSize, PopFit)
            if PopFit[i, j] > PopFit[ix, iy]:
                PopChild[i, j, :] = Pop[ix, iy, :].copy()
            else:
                PopChild[i, j, :] = Pop[i, j, :].copy()
    return PopChild