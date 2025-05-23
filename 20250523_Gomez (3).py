"""
Created on Fri May 23 20:58:23 2025

@author: Alyssa
"""

import numpy as np
import matplotlib.pyplot as plt

#Simulate observed data: Recovery time (in days) of 50 patients
np.random.seed(123)
true_mu = 7  
true_sigma = 1.5
data = np.random.normal(true_mu, true_sigma, 50)

#Prior belief
prior_mu_mean = 8  
prior_mu_precision = 0.8
prior_sigma_alpha = 2
prior_sigma_beta = 1.5

#Posterior update
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

#Draw samples from the posterior
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), 10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, 10000)

#Plot the posterior distributions
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=50, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior distribution of $\mu$ (Recovery Time)')
plt.xlabel('$\mu$ (in days)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=50, density=True, color='magenta', edgecolor='black')
plt.title('Posterior distribution of $\sigma$ (Recovery Time)')
plt.xlabel('$\sigma$ (in days)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

#Posterior statistics
print("Mean of mu (Recovery Time):", np.mean(posterior_mu))
print("Standard deviation of mu (Recovery Time):", np.std(posterior_mu))
print("Mean of sigma (Recovery Time):", np.mean(posterior_sigma))
print("Standard deviation of sigma (Recovery Time):", np.std(posterior_sigma))