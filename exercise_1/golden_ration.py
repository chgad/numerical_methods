
import numpy as np

from  matplotlib import pyplot
# Function to fill a Lis with the delta fo F(n)/F(n-1) - golden ratio


def list_fill(N,precision=float):
    golden = precision((1.0 + np.sqrt(5.0))*0.5)
    plot_list = []
    k,v = 1,1
    l = 0

    while l<N:
        plot_list.append(precision(v/k -golden))
        k,v = v, k+v
        l+=1

    return plot_list

numbers = np.arange(1,1001)

gol_single = list_fill(N=1000,precision=np.float32)
gol_double = list_fill(N=1000)
print(gol_double[7],gol_single[7])
print(type(gol_double[0]))

fig, (ax1,ax2,ax3) = pyplot.subplots(1,3)
ax1.set_title(r"Difference for N in [1,1000]")
ax1.set_xlabel("Number N")
ax1.set_ylabel(r"$\delta(N) -\Phi$")
ax1.plot(numbers,gol_single, 'or', label="single precision")
ax1.plot(numbers,gol_double, 'bs', label="double precision")

ax1.legend(loc="best")
ax2.set_title(r"Zoom for N=8")
ax2.set_xlabel("Number N")
ax2.set_ylabel(r"$\delta(N) -\Phi$")
ax2.plot([numbers[7]],[gol_single[7]], 'or', label="single precision")
ax2.plot([numbers[7]],[gol_double[7]], 'bs', label="double precision")

ax2.legend(loc="best")

ax3.set_title(r"Differnece for 500<N<1000")
ax3.set_xlabel("Number N")
ax3.set_ylabel(r"$\delta(N) -\Phi$")
ax3.plot(numbers[500:],gol_single[500:], 'or', label="single precision")
ax3.plot(numbers[500:],gol_double[500:], 'bs', label="double precision")

ax2.legend(loc="best")

pyplot.show()

