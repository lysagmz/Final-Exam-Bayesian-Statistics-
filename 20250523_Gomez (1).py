"""
Created on Fri May 23 20:58:19 2025

@author: Alyssa
"""

import numpy as np
import matplotlib.pyplot as plt

#Simulate observed data: Number of hours of doing the homework of 30 students
np.random.seed(1)
true_mu = 6.5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, 30)

#Prior belief
prior_mu_mean = 6  
prior_mu_precision = 1  
prior_sigma_alpha = 2  
prior_sigma_beta = 2   

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
plt.hist(posterior_mu, bins=30, density=True, color='pink', edgecolor='black')
plt.title('Posterior distribution of $\mu$ (Homework(No. of Hours))')
plt.xlabel('$\mu$ (Homework(No. of Hours))')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightblue', edgecolor='black')
plt.title('Posterior distribution of $\sigma$ (Homework(No. of Hours))')
plt.xlabel('$\sigma$ (Homework(No. of Hours))')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

#Calculate summary statistics for the posterior distributions
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu (Homework(No. of Hours)):", mean_mu)
print("Standard deviation of mu (Homework(No. of Hours)):", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma (Homework(No. of Hours)):", mean_sigma)
print("Standard deviation of sigma (Homework(No. of Hours))", std_sigma)