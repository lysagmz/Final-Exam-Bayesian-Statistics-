"""
Created on Fri May 23 20:58:22 2025

@author: Alyssa
"""

import numpy as np
import matplotlib.pyplot as plt

#Simulate observed data: Daily Sales (Dollars (in hundreds)) from 30 days
np.random.seed(42)
true_mu = 12  
true_sigma = 3
data = np.random.normal(true_mu, true_sigma, 30)

#Prior belief
prior_mu_mean = 10  
prior_mu_precision = 0.5  
prior_sigma_alpha = 3  
prior_sigma_beta = 3   

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
plt.hist(posterior_mu, bins=30, density=True, color='orange', edgecolor='black')
plt.title('Posterior distribution of $\mu$ (Daily Sales)')
plt.xlabel('$\mu$ (Dollars(in hundreds))')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior distribution of $\sigma$ (Daily Sales)')
plt.xlabel('$\sigma$ (Dollars(in hundreds))')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

#Posterior statistics
print("Mean of mu (Daily Sales):", np.mean(posterior_mu))
print("Standard deviation of mu (Daily Sales):", np.std(posterior_mu))
print("Mean of sigma (Daily Sales):", np.mean(posterior_sigma))
print("Standard deviation of sigma (Daily Sales):", np.std(posterior_sigma))