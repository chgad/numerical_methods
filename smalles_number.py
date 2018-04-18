import numpy as np
print(b'0')
def produce_smalest(precision=None):
    """
    precsion: Class which precision.
    """
    
    y = 2
    if precision:
        x = precision(2.0)
        precision_name = "single"
    else:
        x = np.float64(2.0)
        precision_name = "double"
    
    
    while y*2 == x**-(l-2):
        y = x**-l
        l+=1
    
    print("2^-{} is the smallest number one can produce with {} precision.".format(
        l-1,precision_name))
    


produce_smalest(precision=np.float32) # produce smalest number with single precision
produce_smalest()           # produce smalest number with double precision




