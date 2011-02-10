import numpy as np

N = 2**16 # number of points
r0 = .1         # r0 = sigma**2

v = np.random.normal(0.0, r0**.5, (N,))
print("Assumed noise variance: {0}".format(r0))

# Constant state
x = 1.0
print("Initial state: {0}".format(x))


# State estimate array and initial condition
x_est = np.zeros(N)
x_est[0] = x + (.1*np.random.rand() - .05)
print("Initial state estimate: {0}".format(x_est[0]))

# Error covariance
# Not sure how to choose initial condition for p0
# Should correspond to the mean square error in knowledge of the state
p0 = 0.5
print("Initial state error variance: {0}".format(p0))

for k, xhat in enumerate(x_est[:-1]):
    x_est[k + 1] = x_est[k] + p0/r0/(1.0+p0*k/r0)*(x + v[k] - x_est[k])

import matplotlib.pyplot as plt
k = np.array(range(N))
plt.plot(k, x_est, label=r'$\hat{x}$')
plt.plot(k, x*np.ones(N), label=r'$x$')
plt.xlabel('Sample #')
plt.ylabel('State estimate')
plt.title('Gelb 4.2-1\nNoise variance = {0}, Initial error covariance estimate = {1}'.format(r0, p0))
plt.legend(loc=0)
plt.axis('tight')
plt.show()

"""
filename = "4.2-1.png"
plt.savefig(filename)

# If you want to upload to picassaweb
import picasaAPI
photo = picasaAPI.upload(filename,
        album="Learning about Kalman Filters",
        comment="Kalman filter for x_k+1 = x_k, z_k = x_k + v_k, " +
                "v_k ~ N(0, {0})".format(r0))
"""
