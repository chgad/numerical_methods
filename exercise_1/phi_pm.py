import numpy as np
from matplotlib import pyplot

def raise_phi_directly(N, sign,precision=float):
    if sign =="+":
        phi = precision((-1+np.sqrt(5))*0.5)
    elif sign == "-":
        phi = precision((-1-np.sqrt(5))*0.5)
    else:
        raise Exception

    return phi**N

def phi_by_iteration(N,sign, precision=float):

    l=1
    
    if sign =="+":
        phi = precision((-1+np.sqrt(5))*0.5)
    elif sign == "-":
        phi = precision((-1-np.sqrt(5))*0.5)
    else:
        raise Exception
    result_list = [precision(1.0),phi]

    k,v = precision(1.0), phi

    while l < N:
        k,v = v, k-v
        result_list.append(v)
        l += 1
    return result_list

numbers = np.arange(0,21)

iterative_single = phi_by_iteration(20,sign="+",precision=np.float32)
iterative_double = phi_by_iteration(20,sign="+")

fig , ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2,2)

ax1.plot(numbers,raise_phi_directly(numbers,"+",precision=np.float32),'or', label="Raised directly")
ax1.plot(numbers,iterative_single,"bs", label="Iterative")
ax1.set_title("Single Precsiion")

ax1.legend(loc="best")

ax2.plot(numbers,raise_phi_directly(numbers,"+"),"or", label="Raised directly")
ax2.plot(numbers,iterative_double,"bs", label="Iterative")
ax2.set_title("Double Precsiion")

ax2.legend(loc="best")

ax3.plot(numbers[13],raise_phi_directly(numbers[13],"+",precision=np.float32),'or', label="Raised directly")
ax3.plot(numbers[13],iterative_single[13],"bs", label="Iterative")
ax3.set_title("Single Precsiion Zoomed in")

ax3.legend(loc="best")

ax4.plot(numbers[13],raise_phi_directly(numbers[13],"+"),"or", label="Raised directly")
ax4.plot(numbers[13],iterative_double[13],"bs", label="Iterative")
ax4.set_title("Double Precsiion Zommed in")

ax4.legend(loc="best")


pyplot.show()
# resason for the difference between the iterative and directly raising:
# obviously the directly raising methode has lots of round of errors, probably happening after each multiplikation
# other way could be that addition is more precise than multiplication
