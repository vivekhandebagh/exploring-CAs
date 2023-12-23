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

![big picture](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/caed345e-3cb5-44b9-b835-25a59f372523)

When we are only given the spacetime data, what are the things we can learn? 
1. Using deep learning algorithms, we can infer the look up table that defines the spacetime dynamics of the CA.
2. Use image segmentation analysis to identify the set of distinct global patterns (which I attempted to do using information theoretic feature maps).
3. Use the learned rule and identified structures to analyze the initial conditions that lead to each distinct patterns over time.  

If we let ourselves believe that goal 2 was fully successful, we would have all the pieces of the puzzle. I would have just needed to find way to put those pieces together to achieve goal 3.
The motivation for this project lies in the power of Cellular Automata as a sort of model organism for numerous other complex systems; that if I could achieve these 3 goals with Cellular Automata, I should be able to generalize this process to numerous other complex systems.

# Causal States

At this point, my exploration took a turn into uncovering for myself something more fundamental about the nature of Cellular Automata. 

I was looking hard for a better way to create a segmentation analysis of a CA, and it turned out that my advisor, Prof. James Crutchfield, had worked on a paper that accomplished exactly that. 

["Unsupervised Discovery of Extreme Weather Events Using Universal Representations of Emergent Organization"](https://arxiv.org/abs/2304.12586)

The work proposed an unsupervised segmentation algorithm that classified each point in spacetime with local causal states. Let me explain what that means.

Local Causal States are learned representations that extract organization from spatially-extended dynamical systems.

> In spatiotemporal systems that evolve through local interactions, there is a limit on how fast causal influence can propagate. This limit defines lightcones in the system that are essential features
used in constructing local causal states. The past lightcone of a point in spacetime is the collection of all points at previous times that could possibly have influenced the spacetime point through the local interactions. Similarly, the future lightcone of a spacetime point is the collection of all points at later times that the spacetime point could influence through local interactions.

In other words:

$L^{-}(\vec{r}, t)$ is the set of all possible past lightcones of $(\vec{r}, t)$, and $l^-_i$ is a specific past lightcone realization.

$L^{+}(\vec{r}, t)$ is the set of all possible future lightcones of $(\vec{r}, t)$, and $l^+_i$ is a specific past lightcone realization.

Local Causal Equivalence Relation:
> Two past lightcones are considered causally equivalent if they have the same conditional distribution over co-occuring future lightcones.

$l^-_i$ is causally equivalent to $l^+_j$ if $Pr(L^+ | L^- = l^-_i) = Pr(L^+ | L^- = l^-_j)$.

If $l^-_i$ is causally equivalent to $l^+_j$, we can say that they belong to the same equivalence class.

Then, the set of local causal states is nothing but the set of causal equivalence classes.

What we now have on our hands is a way to assign each point in spacetime with a label that is fundamentally based on the underlying process of the system. In other words, we have a physics-informed unsupervised spacetime segmentation algorithm. The paper demonstrated that this technique actually works using spacetime data of weather events like hurricanes or vortices. These vortices are emergent structures operating on a separate level of organization and the local causal states method is able to identify that. This is exactly what I was trying to achieve with my information theoretic feature maps.

The spacetime data of systems like the weather allow for infinite classes so there are extra steps of clustering and approximation methods that we need not worry for our case with CAs.

I implemented the local causal state method in code (see lightcone.py, local_causal_states.py, and Local_Causal_States_Demo.ipynb), and after applying it I was yet again shocked by the really complex nature of CAs.

I started with a very simple checkerboard CA where I could manually see that every cell is alternating between two causal states.

Here's what I got:
![Checkerboard CA](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/ac23e4c2-5721-4959-ac58-36264aa14eab)

![Causal State Map of Checkerboard CA](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/15b35733-e336-46f6-bd14-071f32256dc7)

We can see that the map correctly classifies the checkerboard into 2 causal classes. Okay, no surprise so far but I want to expand to an idea.

We have a set of local causal states and a method to find them, but how can we understand how the system transitions between these states? 

Well, lets go to the causal state map and lets start by looking at every causal state transition and have a count for the number of times we see each transition. Then, for each transition we can calculate the probability of that transition occuring. We can visualize this by creating a network graph where each node is a causal state and every edge is directed edge that is labelled with the respective transition probability.

Boom! We now have a state machine and we find ourselves chest deep into automata theory.

Here is the state machine for the Checkerboard CA. 

![image](https://github.com/vivekhandebagh/exploring-CAs/assets/54450878/8a482e60-9544-419b-8c47-3dd420302ab6)

In Local_Causal_States_Demo.ipynb I look at the state machines of various CA.
