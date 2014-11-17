import matplotlib.pyplot as plt
import numpy as np
import cPickle

costs_backprop = cPickle.load(open('backprop.pkl', 'r'))['costs']
costs_sgd = cPickle.load(open('target_sgd.pkl', 'r'))['costs']
costs_momentum = cPickle.load(open('target_momentum.pkl', 'r'))['costs']
costs_momentum_no_inverse = cPickle.load(open('target_momentum_no_inverse.pkl', 'r'))['costs']
costs_adagrad = cPickle.load(open('target_adagrad.pkl', 'r'))['costs']

fig_final = plt.figure('Final costs')
ax = fig_final.add_subplot(111)
ax.set_title('Final cost')
final_costs = np.vstack([costs_backprop, costs_sgd[:, 0], costs_momentum[:, 0], costs_adagrad[:, 0]]).T
lines = ax.semilogy(final_costs)
ax.legend(lines, ['backprop', 'target, sgd', 'target, momentum', 'target, adagrad'])
ax.set_xlabel('batches seen')
ax.set_ylabel('$\log(||y-f_1(f_2(f_3(x)))||^2)$')

fig_target_1 = plt.figure('Target 1 costs')
ax = fig_target_1.add_subplot(111)
ax.set_title('Target 1 costs')
target_1_costs = np.vstack([costs_sgd[:, 1], costs_momentum[:, 1], costs_adagrad[:, 1]]).T
lines = ax.semilogy(target_1_costs)
ax.legend(lines, ['target, sgd', 'target, momentum', 'target, adagrad'])
ax.set_xlabel('batches seen')
ax.set_ylabel('$\log(||h_1-\hat{h_1}||^2)$')

fig_target_2 = plt.figure('Target 2 costs')
ax = fig_target_2.add_subplot(111)
ax.set_title('Target 2 costs')
target_2_costs = np.vstack([costs_sgd[:, 2], costs_momentum[:, 2], costs_adagrad[:, 2]]).T
lines = ax.semilogy(target_2_costs)
ax.legend(lines, ['target, sgd', 'target, momentum', 'target, adagrad'])
ax.set_xlabel('batches seen')
ax.set_ylabel('$\log(||h_2-\hat{h_2}||^2)$')

fig_inverse_cost = plt.figure('Inverse costs')
ax = fig_inverse_cost.add_subplot(111)
ax.set_title('Inverse costs')
inverse_costs = np.vstack([costs_sgd[:, 3], costs_momentum[:, 3], costs_adagrad[:, 3]]).T
lines = ax.semilogy(inverse_costs)
ax.legend(lines, ['target, sgd', 'target, momentum', 'target, adagrad'])
ax.set_xlabel('batches seen')
ax.set_ylabel('$\log(||\hat{h_2}-f_2(g_2(\hat{h_2}))||^2)$')

fig_no_inverse = plt.figure('Learning inverse')
ax = fig_no_inverse.add_subplot(111)
ax.set_title('Final cost')
final_costs = np.vstack([costs_momentum[:, 0], costs_momentum_no_inverse[:, 0]]).T
lines = ax.semilogy(final_costs)
ax.legend(lines, ['target w/ inverse leanring', 'target w/o inverse learning'])
ax.set_xlabel('batches seen')
ax.set_ylabel('$\log(||y-f_1(f_2(f_3(x)))||^2)$')
plt.show()
