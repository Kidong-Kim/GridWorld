# GridWorld - Reinforcement Learning

## Abstract

- [Project for CS188](http://inst.eecs.berkeley.edu/~cs188/sp20/project3/) - "Introduction to Artificial Intelligence" at UC Berkeley during Fall 2021.
- Modified valueIterationAgents.py, qlearningAgents.py, analysis.py to implement Value Iteration & Standard Prioritized Sweeping Algorithm for offine agentand Q-learning & Epsilon Greedy Action Selection for reinforcement learning agent.
- Created a classic version of the GridWorld that displays rewards for policies at each state on the grid
- The code is based on skeleton code from the class. Project was completed using the PyCharm Python IDE. This submission received full score.

## Project description

---

 These agents are able to learn optimal policies until convergence when put into an environment that can be characterized as a Markov Decision Process using Transition functions (non-deterministic acting) and Reward functions.

Agent (blue dot) learns optimal actions at each square in GridWorld by running Q-Learning in MDP setting.

![188_p3a](https://user-images.githubusercontent.com/54779918/83335690-be5d0d80-a2ae-11ea-93ce-13f9e2c3a3b6.gif)


Crawler learns to crawl along positive x axis using Q-Learning.

![188_p3b](https://user-images.githubusercontent.com/54779918/83335701-d3d23780-a2ae-11ea-86ca-f3c5943df161.gif)

Pacman learns great strategies through approximate Q-learning in feature representation of game states.

![188_p3c](https://user-images.githubusercontent.com/54779918/83335925-60312a00-a2b0-11ea-86cb-01cd2716828c.gif)

