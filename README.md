# HopPics : Python Implementation of Binary Hopfield Network for self-reconstructing binary images

[Github Repository](https://github.com/lenoctambule/HopPics)

## Installation

Installing requirements :
```console
$ py -m pip install -r requirements.txt
```
## Usage

Running tests :
```console
$ py test.py <image_path> <nsteps>
$ py test.py example.png 4
```

Usage in code :
```py
from HopPics import *

hp = HopPics('path_to_pic.png')
hp.reconstruct_from_noise()
```

## I. Introduction

Hopfield networks (HNs) are among the simplest neural networks models to understand and explain. It only requires basic understanding of matrices and graph theory. It tackles the notion of associative memory and shows a few limitations that are worth taking note of.

This project is a simple implementation of Binary Hopfield Networks to make images that are able to repair themselves through pattern recognition.

## II. How it works

### A. Training

$$ K = (B,W) $$

We chose to represent our model with a complete graph K that consists in a set of weights $W$ and a set of nodes $B$. Each node $ i \in B $ represents a bit that can either hold the value -1 or 1 and the weights are calculated through a training method that I chose to be the Hebbian rule ("Neurons that fire together, wire together"). This is the formula for that method :

$$ w_{i,j} = {1 \over |B|} \sum_{ p \in P } b_i^p b_j^p $$

Where $ w_{ij} $ is the weight of the edge that connects the nodes $i$ and $j$, $ P $ is the set of patterns that we wish to train the network to recognize, $ b_i^p $ is the value of the node $i$ in the pattern $p$ where in our case $ b_i^p \in \\\{1,-1\\\} $ and finally $|B|$ is the order of our graph (number of nodes).

What this does is basically making the weights decrease if $ b_i^p \ne b_j^p $ because if $ b_i^p, b_j^p \in \\\{1,-1\\\} $ then $ b_i^p * b_j^p = -1 * 1 = -1 $. And if they are equal (firing together) then $ b_i^p * b_j^p = (-1 * -1) \text{ or } (1 * 1) = 1 $. So the weights are "strenghtened" in the case that two neurons are firing at the same time.

### B. Associative memory

To calculate each step, we do the following calculations.

$$ b_i = \sum_{j \in B, j \ne i} b_j w_{i,j} $$

And then, each bit goes through an activation function that I chose to be :

$$ \phi(b_i) = sgn(b_i) $$

The fact that the weights are strenghtened if they have the same state will give us what we can call associative memory or pattern recognition. This is because the result of each multiplication between the weight $w_{i,j}$ and the value of each bit $b_j$ in the weighted sum above can pull the value of $ b_i $ either towards 1 or -1, and if enough bits  $ b_j $ in this weighted sum match the trained pattern $b_j^p$ it will be pulled towards $b_i^p$ because of the way we trained our network in II.A.

## III. Examples

<figure>
<p align=center>
<img src=./figures/margaret_hamilton.png>
<figcaption><p align=center><b>Fig 2</b> : Examples of a destroyed picture of Margaret Hamilton reconstructed after 4 steps. </p></figcaption>
</p>
</figure>

<figure>
<p align=center>
<img src=./figures/panda.png>
<figcaption><p align=center><b>Fig 3</b> : Examples of a destroyed image of a panda reconstructed after 8 steps. </p></figcaption>
</p>
</figure>

Both of these examples have been achieved by first destroying the image by adding random noise to the image (if you squint hard enough you can still see M. Hamilton's face). And even though those image are almost unrecognizable they are still able to snap back to the original.

## IV. Conclusion

Hopfield Networks are a great entry point to learning AI and even though HNs are old they might still be useful and interesting to study. More precisely the limitations of HNs are the ones that are worth taking note of as it extends to other models of NNs as well and can help designing other models given a set of constraints and requirements.

The main limitation of HNs is the limited number of patterns that can be trained for the same model. But another limitation is the possibility for the network to settle for a pattern $p$ that hasn't been trained ($p \notin P$). Also the fact that HNs are recurrent means that they offer little possibility of parallelization.

Recent work has shown that this behaviour of pattern recognition and self-reconstruction can also be achieved through the more convenient method of training cellular automatas to self regenerate or self classification. And I think this is where I'll be headed towards for my next AI project.

## Links

[Github Repository](https://github.com/lenoctambule/HopPics)

## Useful Ressources

- [*Hopfield Network*, Alice Julien-Laferriere](http://perso.ens-lyon.fr/eric.thierry/Graphes2010/alice-julien-laferriere.pdf)
- [*Hopfield network*, John J. Hopfield, Scholarpedia](http://www.scholarpedia.org/article/Hopfield_network)
- [Randazzo, et al., "Self-classifying MNIST Digits", Distill, 2020.](https://distill.pub/2020/selforg/mnist/)
- [Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.](https://distill.pub/2020/growing-ca/)
