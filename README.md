# exploring-CAs

![Elementary CA 225](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/a776d7e2-9773-4d9b-9690-52b260891487)

# Motivation
This project served as a means to investigate Cellular Automata, the beautiful objects made of nothing but simple 1s and 0s.

Cellular automata (CA) are mathematical models that simulate the behavior of complex systems that operate through simple local interactions. Pioneered in the 1940s by the mathematician John von Neumann, CAs now provide a simple framework that can be used to understand the nature and properties of complex systems and networks. 

The basic arrangement of a CA is a lattice of cells, where each cell can occupy one of a finite number of $\textit{k}$ states. At each time step, the state of each cell is updated according to a function or a set of transition rules that considers the current state of the cell and its neighborhood of size $\textit{r}$. Most cellular automata that have been studied are 1-dimensional and binary. The state of a cellular automata at a given time step, then, is given by the binary string that represents the values of each cell and the state space is the set of all possible binary strings of length $\textit{L}$

<img width="611" alt="Screenshot 2023-10-14 141406" src="https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/ccf47831-53bd-4cd8-8a4d-a21f9ad381c5">

There is no mystery in the underlying dynamics of a CA, yet there is still a surprising sense of higher order patterns. There are some _structures_ that seem to propagate through out space and time, some that simply oscillate between a set of states, and some that remain fixed through time. The motivation behind my exploration was to try to find a way to identify the set of distinct emergent structures that arise in the CA spacetime dynamics, and to identify the set of initial conditions that map to each distinct structure. 

To achieve this, I had supposed that machine learning would be helpful (Note that I had no previous ML background before this project).

# Rule Discovery
I started with the simple task of having a model learn the lookup table for complex CAs. Essentially, I tried to input a length $\textit{L}$ binary string to a Neural Network, and expected that a trained model would ouput the correct length $\textit{L}$ binary string that represents the CA state in the next time step. 

Result: 
A simple feed-forward network was simply unable to approximate the look-up table of a complicated CA (See ANN_Rule_Discovery_1d.ipynb). 
As to why such a task is difficult for a neural network is an intriguing mathematical problem whose answer I do not have.

After furthering my knowledge of different deep learning algorithms, I saw a connection between Convolutional Neural Networks (CNNs) and CAs. In CNNs, the convolution operation performed on a two-dimensional image specifically resembles a CA update rule operating on a two-dimensional lattice

![Convolution](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/cf7773cd-3790-447b-a854-068026e10b4b)

This connection has been explored, and a [study was done that developed and trained a CNN model to learn the rule set of arbitrary CAs](https://arxiv.org/abs/1809.02942). In the study, training successfully converged for all studied CA rule-sets. However, it’s important to note that this performance should not be interpreted as its capability to generalize the learning, as the networks were trained to learn all possible inputs of the CA. Thus, I set off trying to see if I could get a CNN to generalize its learning of CA. 

Result:
A sufficiently overcomplete CNN is able to learn the look-u table of complicated CAs (both 1D and 2D, see CNN_Rule_Discovery_1D.ipynb and CNN_Rule_Discovery_2D.ipynb). 
This [paper](https://arxiv.org/abs/2009.01398) helped quite a bit.

Yay! I think everyone should start celebrating their first model converging like birthdays! 

Okay, what now?

Next, I tried to tackle the  goal of devising some sort of unsupervised semantic segmentation algorithm to distinguish different structures in the spacetime image. The method I thought of was to consider a local window in the CA spacetime and calculate the values of information theoretic measures in that local window. If I do this operation on a window for each cell in the spacetime, it will result in a feature map that will quantify the different patterns or structures that appear throughout the spacetime. However, once I had reached this point, I realized I was stuck. I saw that I did not have a way to exactly check if my information theoretic feature map is correctly identifying the set of distinct structures. 

This is when I found the paper, "Physics-Informed Representation Learning for Emergent Organization in Complex Dynamical Systems", which proposed an unsupervised segmentation algorithm that classified each point in spacetime with local causal states. I implemented the ideas of lightcone construction and local causal equivalence in code and was successfully able to apply it to my interest in CA dynamics. Doing this was the most valuable part of doing this project, as initially, I had not fully understood how these Hidden Markov Models and epsilon-machines applied to real-world systems (simply because the only processes I really analyzed were very simple ones like the Golden Mean Process). However, when I applied it to a system I was studying, everything I learned in the two quarters of classes I took finally clicked. Now, I feel that I have the greater understanding that I need to really dive into systems that are more complicated than Cellular Automata.
