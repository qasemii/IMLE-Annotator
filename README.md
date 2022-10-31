# iMLE-Annotator

## Introduction test

Implicit MLE (I-MLE) makes it possible to include discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or integer linear program (ILP) solvers, in standard deep learning architectures. The core idea of I-MLE is that it defines an *implicit* maximum likelihood objective whose gradients are used to update upstream parameters of the model. Every instance of I-MLE requires two ingredients:
1. A method to approximately sample from a complex and intractable distribution induced by the combinatorial solver over the space of solutions, where optimal solutions have the highest probability mass. For this, we use Perturb-and-MAP (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand.
2. A method to compute a surrogate empirical distribution: Vanilla MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to an empirical distribution, we have to design surrogate empirical distributions. Here we propose two families of surrogate distributions which are widely applicable and work well in practice.
