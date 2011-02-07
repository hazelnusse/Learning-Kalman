import numpy as np

N = 4096         # number of points
r0 = .01         # r0 = sigma**2
v = np.random.normal(0.0, r0**.5, (N,))

# Constant state
x = 1.0

# State estimate array and initial condition
x_est = np.zeros(N)
x_est[0] = x + np.random.rand()

# Error covariance
# Not sure how to choose initial condition for p0
p0 = 0.01 #np.random.rand()

for k, xhat in enumerate(x_est[:-1]):
    x_est[k + 1] = x_est[k] + p0/r0/(1.0+p0*k/r0)*(x + v[k] - x_est[k])

import matplotlib.pyplot as plt
k = np.array(range(N))
plt.plot(k[:500], x_est[:500])
plt.plot(k[:500], x*np.ones(500))
plt.xlabel('Sample #')
plt.ylabel('State estimate')
plt.title('Initial Variance = {0},\n Initial Error covariance estimate = {1}'.format(r0, p0))
plt.axis('tight')
plt.show()
