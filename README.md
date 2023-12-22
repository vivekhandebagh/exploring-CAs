# exploring-CAs

![Elementary CA 225](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/a776d7e2-9773-4d9b-9690-52b260891487)

# Motivation
This project served as a means to investigate Cellular Automata, the beautiful objects made of nothing but simple 1s and 0s.

Cellular automata (CA) are mathematical models that simulate the behavior of complex systems that operate through simple local interactions. Pioneered in the 1940s by the mathematician John von Neumann, CAs now provide a simple framework that can be used to understand the nature and properties of complex systems and networks. 

The basic arrangement of a CA is a lattice of cells, where each cell can occupy one of a finite number of $\textit{k}$ states. At each time step, the state of each cell is updated according to a function or a set of transition rules that considers the current state of the cell and its neighborhood of size $\textit{r}$. Most cellular automata that have been studied are 1-dimensional and binary. The state of a cellular automata at a given time step, then, is given by the binary string that represents the values of each cell and the state space is the set of all possible binary strings of length $\textit{L}$

<img width="611" alt="Screenshot 2023-10-14 141406" src="https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/ccf47831-53bd-4cd8-8a4d-a21f9ad381c5">

There is no mystery in the underlying dynamics of a CA, yet there is still a surprising sense of higher order patterns. There are some _structures_ (interchangeable with 'patterns') that seem to propagate through out space and time, some that simply oscillate between a set of states, and some that remain fixed through time. The motivation behind my exploration was to try to find a way to identify the set of distinct emergent structures that arise in the CA spacetime dynamics, and to identify the set of initial conditions that map to each distinct structure. 

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

# Information Theoretic Feature Maps
If we look at the spacetime of CA 225 (shown in the image at the top), we can visually distinguish a couple separate types of global structures. Now, my aim was to find a way to distinguish different structures in the spacetime image. The method I thought of was to consider a local window in the CA spacetime and calculate the values of $H$ and $I$ in that local window so that we can characterize the complexity of local patterns. If we do this operation on a window for each cell in the spacetime, it will result in a feature map that will illuminate the different patterns or structures that appear throughout the spacetime. 

![spacetime image](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/58f177db-77e9-4790-8bf9-513fc595bd62)
![entropy map](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/9223a854-50b3-4e08-9376-5a12c113f485)
![mutual info map](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/dd175242-6390-42b9-8f26-7cfc98f13079)
![added](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/d9692c48-304b-45ce-ac6a-6a514f539025)

Note that nothing is stopping us to add other feature maps like a 1's density map (count up number of 1s in a neighborhood and assign it to center cell) or other information theoretic measures. 

I played around with the idea of a neural network reproducing the spacetime image given the set of feature maps (see InfoMap_to_State.ipynb). That would then result in a mapping between a combination of feature values to a particular CA structure, which is represented by the trained network. Then, if we input a single combination of feature values, the network should output a distinct spacetime structure. However, it didn't work out so simply.

Possible reasons and limitations:
1. Shannon Entropy, Mutual Information, and other similar information theoretic measures are symmetric, meaning that the measure doesn't distinguish between all 1s and all 0s.
2. a spacetime structure is influenced by cells outside the boundary of the local window that feature map considered
3. a spacetime structure is likely characterized by multiple combinations of feature values that also need to be organized in a certain way and looking at a simple window and assigning a value to it might not be sufficient.

There might have been scope to put further effort down this path but I didn't think it was worth the time.

# Big Picture
In terms of the bigger picture, here's the line of thought I was treading on: 

![big picture](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/9ed38888-6829-47fa-b2ae-34cbd7b82a44)

When we are only given the spacetime data, what are the things we can learn? 
1. Using deep learning algorithms, we can infer the look up table that defines the spacetime dynamics of the CA.
2. Use image segmentation analysis to identify the set of distinct global patterns (which I attempted to do using information theoretic feature maps).
3. Use the learned rule and identified structures to analyze the initial conditions that lead to each distinct patterns over time.  

If we let ourselves believe that goal 2 was fully successful, we would have all the pieces of the puzzle. I would have just needed to find way to put those pieces together to achieve goal 3.
The motivation for this project lies in the power of Cellular Automata as a sort of model organism for numerous other complex systems; that if I could achieve these 3 goals with Cellular Automata, I should be able to generalize this process to numerous other complex systems.

# Causal States

This is when I found the paper, "Physics-Informed Representation Learning for Emergent Organization in Complex Dynamical Systems", which proposed an unsupervised segmentation algorithm that classified each point in spacetime with local causal states. I implemented the ideas of lightcone construction and local causal equivalence in code and was successfully able to apply it to my interest in CA dynamics. Doing this was the most valuable part of doing this project, as initially, I had not fully understood how these Hidden Markov Models and epsilon-machines applied to real-world systems (simply because the only processes I really analyzed were very simple ones like the Golden Mean Process). However, when I applied it to a system I was studying, everything I learned in the two quarters of classes I took finally clicked. Now, I feel that I have the greater understanding that I need to really dive into systems that are more complicated than Cellular Automata.
