This repo try to test some ideas inspired from the target prop setting and is inspired by an implementation from Dong-Hyun Lee

The setting of the two experiments is very simple. A mlp with two hidden layers in dimension 3 tries to approximate one periode of a noisy sinusoide.

backprop_sinus_approx.py optimizes this mlp thanks to a classical backpropagation.
target_sinus_approx.py optimizes this mlp thanks to target propagation.

The learning rate rule can be chosen by commenting/uncommenting the right lines
