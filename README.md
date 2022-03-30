# `schnax`: SchNet in JAX and JAX-MD
This is a re-implementation of the `SchNet` neural network architecture in [JAX](https://github.com/google/jax), [haiku](https://github.com/deepmind/dm-haiku), and [JAX-MD](https://github.com/google/jax-md).
`schnax` is intended as a drop-in replacement for the original `pytorch` implementation, allowing the use of trained weights obtained with [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack).


## References
* [1] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.  
*SchNet - a deep learning architecture for molecules and materials.* 
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [3] K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller. *SchNetPack: A Deep Learning Toolbox For Atomistic Systems.* J. Chem. Theory Comput. [10.1021/acs.jctc.8b00908](https://doi.org/10.1021/acs.jctc.8b00908) [arXiv:1809.01072.](https://arxiv.org/abs/1809.01072v1) (2018)
