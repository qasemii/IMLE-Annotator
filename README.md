# iMLE-Annotator

## Introduction

Implicit MLE (I-MLE) makes it possible to include discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or integer linear program (ILP) solvers, in standard deep learning architectures. The core idea of I-MLE is that it defines an *implicit* maximum likelihood objective whose gradients are used to update the upstream parameters of the model. Every instance of I-MLE requires two ingredients:
1. A method to approximately sample from a complex and intractable distribution induced by the combinatorial solver over the space of solutions, where optimal solutions have the highest probability mass. For this, we use Perturb-and-MAP (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand.
2. A method to compute a surrogate empirical distribution: Vanilla MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to an empirical distribution, we have to design surrogate empirical distributions. Here we propose two families of surrogate distributions that are widely applicable and work well in practice.

The picture below shows an output of our model on the BeerAdocate dataset for Aroma
. The model (in this example) is supposed to select 5 samples from the given input. Each color means as follows:
- Red: Selected by Machine 
- Blue: Selected by Human
- Purple: Selected by Machine and Human

So (Red and purple) are machine reasoning and (Blue and purple) are human reasoning.

![image](https://github.com/qasemii/imle-annotator/blob/main/images/BeerAvocate-Aroma-K5.png)

Note: highlights are not provided during the training and they are just used for testing model performance.

## Run

To train the baseline model on the BeerAdvocate use the following command:
```bash
python3 beer-cli.py \
-a 0 -e 1 -b 40 -k 3 -H 250 -m 350 -K 10 -r 1 -M imle \
--imle-samples 1 --imle-noise gumbel --imle-lambda 1000.0\
--imle-input-temperature 1.0 --imle-output-temperature 1.0
```

To train the model on the e-SNLI use the following command:
```bash
python3 esnli-cli.py\
-e 10 -b 256 -k 3 -H 250 -m 150 -K 3 -r 1 -M imle --highlight False\
--imle-samples 1 --imle-noise gumbel --imle-lambda 1000.0\
--imle-input-temperature 1.0 --imle-output-temperature 1.0
```

The following command can be used for to train the Bert-to-Bert model for the BeerAdvocate dataset:
```bash
python3 beer-bert.py \
-a 0 -e 10 -b 128 -m 350 -K 10 -r 1 -M imle -B prajjwal1/bert-mini\
--imle-samples 1 --imle-noise gumbel --imle-lambda 1000.0\
--imle-input-temperature 1.0 --imle-output-temperature 1.0
```
