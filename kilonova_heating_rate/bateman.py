import numpy as np
xcrit = np.inf#38.0
def calc_M0_1(x):
    if x<xcrit:
        return (1.0-np.exp(-x))/x
    else:
        return 1.0/x

def calc_M0_2(x1, x2):
    if x2<xcrit:
        return  (calc_M0_1(x2)-calc_M0_1(x1))/(x1-x2)
    else:
        return 1.0/(x1*x2)
    
#x1 < x2 <x3
def calc_M0_3(x1, x2, x3):
    return (calc_M0_2(x1,x2) - np.exp(-x1)*calc_M0_2(x2-x1,x3-x1))/x3

def calc_M0_4(x1, x2, x3, x4):
    return (calc_M0_3(x1,x2,x3) - calc_M0_3(x2,x3,x4))/(x4-x1)

def calc_M0_5(x1, x2, x3, x4, x5):
    return (calc_M0_4(x1,x2,x3,x4) - calc_M0_4(x2,x3,x4,x5))/(x5-x1)

def calc_M0_6(x1, x2, x3, x4, x5, x6):
    return (calc_M0_5(x1,x2,x3,x4,x5) - calc_M0_5(x2,x3,x4,x5,x6))/(x6-x1)

def calc_M0_7(x1, x2, x3, x4, x5, x6, x7):
    return (calc_M0_6(x1,x2,x3,x4,x5,x6) - calc_M0_6(x2,x3,x4,x5,x6,x7))/(x7-x1)

def calc_M0_8(x1, x2, x3, x4, x5, x6, x7, x8):
    return (calc_M0_7(x1,x2,x3,x4,x5,x6,x7) - calc_M0_7(x2,x3,x4,x5,x6,x7,x8))/(x8-x1)

def calc_M0_9(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (calc_M0_8(x1,x2,x3,x4,x5,x6,x7,x8) - calc_M0_8(x2,x3,x4,x5,x6,x7,x8,x9))/(x9-x1)

def calc_M0_10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (calc_M0_8(x1,x2,x3,x4,x5,x6,x7,x8,x9) - calc_M0_9(x2,x3,x4,x5,x6,x7,x8,x9,x10))/(x10-x1)
