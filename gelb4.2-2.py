import numpy as np
from sympy import Matrix, symbols, S

H = np.array([0, 1])
s1, s12, s2, r2 = symbols('s1 s12 s2 r2')

Pk_pre = Matrix([[s1, s12],[s12, s2]])
Hk = Matrix(H)
print("Pk(-) =\n" + str(Pk_pre))
print("Hk = " + str(Hk))

Kk = (Pk_pre.multiply(Hk.transpose()))*(S(1)/
        ((Hk.multiply(Pk_pre)).multiply(Hk.transpose())[0] + r2))
print("Kalman gain:\n" + str(Kk))
Pk_post = (Matrix([[1,0], [0,1]]) - Kk.multiply(Hk)).multiply(Pk_pre)
print("Pk(+) =\n"+str(Pk_post.expand()))

N = 2**10        # number of points
r2 = .1         # variance r2 = sigma**2
v = np.random.normal(0.0, r2**.5, (N,))

# True state, constant
x = np.ones((N, 2))
x[:, 0] *= .5
x[:, 1] *= .25
print("Initial state: x[0] =\n{0}".format(x[0]))

# Estimated state, initially within 5% of true state
x_est = np.zeros((N, 2))
x_est[0] = x[0] + (np.random.rand(2)*.1 - .05)*x[0]  # really x_est_0_post
print("A posteriori initial state estimate: x_est_0(+) =\n{0}".format(x_est[0]))

# Estimated error covariance matrix
P = np.zeros((N, 2, 2))
sigma1 = 0.53
sigma2 = 0.02
sigma12 = 0.3
P[0] = np.array([[sigma1, sigma12],[sigma12, sigma2]])
print("A posteriori initial error covariance: P_0(+) =\n{0}".format(P[0]))

for k in range(1, N):

    # Extrapolation to next time step
    x_est_k_pre = x[k - 1]      # state transition is identity
    Pk_pre = P[k - 1]      # no process noise

    # Measurement update
    zk = x[k, 1] + v[k]
    # Kalman gain
    Kk = Pk_pre.dot(H.transpose()) / ((H.dot(Pk_pre)).dot(H.transpose()) + r2)

    # A posteriori state estimate update
    x_est[k] = x_est_k_pre + Kk.dot(zk - H.dot(x_est_k_pre))

    # A posteriori covariance update
    P[k, 0, 0] = P[k-1, 0, 0] - P[k-1, 0, 1]**2.0/(r2 + P[k-1, 1, 1])
    P[k, 0, 1] = P[k-1, 0, 1] - P[k-1, 0, 1]*P[k-1, 1, 1]/(r2 + P[k-1, 1, 1])
    P[k, 1, 0] = P[k-1, 0, 1] - P[k-1, 0, 1]*P[k-1, 1, 1]/(r2 + P[k-1, 1, 1])
    P[k, 1, 1] = P[k-1, 1, 1] - P[k-1, 1, 1]**2.0/(r2 + P[k-1, 1, 1])

print("Final error covariance:\n{0}".format(P[-1]))


import matplotlib.pyplot as plt
k = np.array(range(N))
plt.plot(k, x_est[:,0], 'b-', label='$\hat{x}_1$')
plt.plot(k, x[:,0], 'r--', label='$x_1$')
plt.plot(k, x_est[:,1], 'g-', label='$\hat{x}_2$')
plt.plot(k, x[:,1], 'k--', label='$x_2$')
plt.xlabel('Sample #')
plt.ylabel('State estimate')
plt.title('Gelb Problem 4.2-2, $x_2$ measurement noise variance = {0}\n'.format(r2)
          + 'Initial error covariance $(\sigma_1, \sigma_2, \sigma_{12}) = ' +
          '({0}, {1}, {2})$'.format(P[0, 0, 0], P[0, 1, 1], P[0, 0, 1]))
plt.legend(loc=0)
plt.axis('tight')
plt.show()

"""
filename = "4.2-2.png"
plt.savefig(filename)

# If you want to upload to picassaweb
import picasaAPI
photo = picasaAPI.upload(filename,
        album="Learning about Kalman Filters",
        comment="Kalman filter for x_k+1 = x_k, z_k = x_k + v_k, " +
                "v_k ~ N(0, {0})\nx is 2-d, using a noisy measurement of x2 to "
                "estimate the state of x1.".format(r2))
"""
