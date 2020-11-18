# DQN Pong Experiment

Deep Q Learning to play pong by reading the pixels on screen.

![Intro](https://i.imgur.com/Xucaas1.gif)

## Requirements

numpy == 1.17.3

opencv-python >= 4.1.0

tensorflow >= 2.3.0

tensorboard >= 2.3.0

``` pip3 install -U -r requirements.txt ```

## Usage

Install the requirements mentioned above.


### Training
1. Set TRAIN to True on top of DQNPongExperiment.py

2. Run DQNPongExperiment.py

```python DQNPongExperiment.py```

3. Run tensorboard to observe the results

```tensorboard --logdir=logs```

![Tensorboard](https://i.imgur.com/dqr2HfB.jpg)

4. Play by changing the forementioned TRAIN-variable to False and run
the DQNPongExperiment.py again

## Results

Training of this model is awfully slow. The agent seems to learn 
the basics of the game after around 150 games (takes around a night worth of training
on 4770k and 1070gtx on 50x50 input size), but will still be far from fully converging.  
Possible improvements to this problem: increasing the input size since alot of information
is lost with the current resolution of 50x50 and even worse is that the ball itself will
sometimes be resized out. Also increasing the epsilon mid-training could provide the agent
new perspectives of the game. Last but not least, Prioritized Experience Replay could
show a significant improvement: https://arxiv.org/abs/1511.05952


## Acknowledgments

* https://www.youtube.com/watch?v=7Lwf-BoIu3M

