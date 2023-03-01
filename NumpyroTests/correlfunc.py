'''
Storage space for the general correlation function of a DRW / tophat tfer func
'''

import numpy as np

def correlfunc(du,dw,w):

    if du<dw:
        out = w-dw + (np.exp(-dw)-np.exp(-w)) * np.cosh(du)
    elif du<w:
        out = (w-du) * np.exp(-w) * np.cosh(du) * np.exp(-du) * np.cosh(dw)
    else:
        out = np.exp(-du) * (np.cosh(w) - np.cosh(dw))

    #norm_coeff
    out /= (w-dw) * (np.exp(-w) - np.exp(-dw))

    return(out)


def correlfunc_origvar(t1,t2,l1,l2,w1,w2,tau):

    dw = abs(w1-w2)/2 /tau
    w  = (w1+w2)/2 / tau
    du = abs((t1-l1)-(t2-l2)) / tau

    return(correlfunc(du,dw,w))