"""_NOTE_
You need to complete those part denoted by "???"
"""
import numpy as np
from scipy.stats import norm # what function should be imported to create a normal distribution?
import matplotlib.pyplot as plt

mu_set1 = 0
sigma1 = 0.1
sigma2 = 0.3
sigma3 = 0.7

sigma_set2 = 0.5
mu1 = -5
mu2 = 0
mu3 = 5

x = np.linspace(-10, 10, num=1000)

dist1 = norm(mu_set1, sigma1)
dist2 = norm(mu_set1, sigma2)
dist3 = norm(mu_set1, sigma3)

dist4 = norm(mu1, sigma_set2)
dist5 = norm(mu2, sigma_set2)
dist6 = norm(mu3, sigma_set2)

dist1_pdf = dist1.pdf(x)#What function should be used to calculate the "probability density"?
dist2_pdf = dist2.pdf(x)
dist3_pdf = dist3.pdf(x)
dist4_pdf = dist4.pdf(x)
dist5_pdf = dist5.pdf(x)
dist6_pdf = dist6.pdf(x)

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.set_ylabel("probability")
ax1.set_xlabel("x")
ax1.plot(x, dist1_pdf, 'r', label="mu=0, sigma=0.1")
ax1.plot(x, dist2_pdf, 'g', label="mu=0, sigma=0.3")
ax1.plot(x, dist3_pdf, 'b', label="mu=0, sigma=0.7")
ax1.set_title(label='Set 1')
ax1.legend()

ax2.set_ylabel("probability")
ax2.set_xlabel("x")
ax2.plot(x, dist4_pdf, 'r', label="mu=-5, sigma=0.5")
ax2.plot(x, dist5_pdf, 'g', label="mu=0, sigma=0.5")
ax2.plot(x, dist6_pdf, 'b', label="mu=5, sigma=0.5")
ax2.set_title(label='Set 2')
ax2.legend()

plt.show()


### Briefly explain your understanding hereunder.
"""
In set 1, we have same mu value for 3 distributions, so they will have the center or expectation value at mu value. The larger sigma value will lead to larger range of value and less likely for mu value.
In set 2, we have same sigma value so we can see 3 distribution has the same value range but because they have different mu value so there centers are at different places.
"""