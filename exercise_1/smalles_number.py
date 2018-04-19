import numpy as np
print(b'0')
def produce_smalest(precision=float):
    """
    precsion: Class which precision.
    return: smallest exponent of 2 wich can be represented
    """
    l=1
    y = 2
    x=2.0  
    while y>0.0:
        y = precision(x**-l)
        l+=1
    
    return l-2
        

exponent = produce_smalest(precision=np.float32) # produce smalest number 
                                                 # with single precision
print("2^-{} is the samllest number which can be represented using single precision".format(exponent), np.float32(2**-exponent))
exponent=produce_smalest()           # produce smalest number with double precision
print("2^-{} is the samllest number which can be represented using double precision".format(exponent), 2.0**-exponent)



