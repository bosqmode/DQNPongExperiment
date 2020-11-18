from DQNAgent import DQN
import numpy as np 
import gym
import gym.spaces
import cv2

ACTIONS = 3 
MAX_EPISODES = 99999999
TRAIN = True
WEIGHTS_NAME = "PONGW"

INPUT_SHAPE = (50,50,1)
CROPPING = (30, 45)

EXPREPLAY_CAPACITY = 20000
OBSERVEPERIOD = 100000
START_EPSILON = 1
MIN_EPSILON = 0.05
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_DECAY = 0.000005
TARGET_UPDATE_GAME_INTERVAL = 1

class PongExperiment:
    def __init__(self):
        self.agent = DQN(num_actions=ACTIONS, input_shape=INPUT_SHAPE, gamma = GAMMA, expreplaycap = EXPREPLAY_CAPACITY, batchsize = BATCH_SIZE, startepsilon = START_EPSILON, minepsilon = MIN_EPSILON, epsilondecay = EPSILON_DECAY, obsperiod=OBSERVEPERIOD)
        self.env = gym.make('Pong-v0').unwrapped
        self.render = False

    def ProcessFrame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = img[CROPPING[0]:CROPPING[0]+img.shape[0]-CROPPING[1], 0:img.shape[1]-1]
        img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        _,img = cv2.threshold(img,90,255,cv2.THRESH_BINARY)
        
        if self.lastimg is None:
            self.lastimg = img

        img = cv2.addWeighted(img,1,self.lastimg,0.6,0)
        self.lastimg = img
        
        if self.render:
            cv2.imshow('Input',cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(20)

        return (img/255)

    def TrainEpisode(self, train):
        self.lastimg = None
        state = self.ProcessFrame(self.env.reset())
        score = 0
        while True:
            action = self.agent.EpsilonGreedyAction(state)
            newstate, reward, done, _ = self.env.step(action+1) 
            nextstate = self.ProcessFrame(newstate)
            score += reward

            self.agent.AddSample(state,action,reward,nextstate,done)

            if done >= 1:
                break

            if train:
                self.agent.Train()
            state = nextstate
        return score

    def PlayEpisode(self, train):
        self.lastimg = None
        state = self.ProcessFrame(self.env.reset())
        score = 0
        while True:
            action = self.agent.PredictAction(state)
            newstate, reward, done, _ = self.env.step(action+1) 
            nextstate = self.ProcessFrame(newstate)
            score += reward

            if done >= 1:
                break

            state = nextstate

            if self.render:
                self.env.render()
        return score

    def Average(self, episodes):
        avg = 0
        for _ in range(episodes):
            s = self.TrainEpisode(False)
            avg += s
        avg = avg/float(episodes)
        return avg

    def Train(self):
        print("Starting training...")
        self.render = False
        for e in range(MAX_EPISODES):
            episode_score = self.TrainEpisode(True)
            self.agent.lastscore = episode_score

            print("Game: {0}, Score: {1:.2f}".format(e, episode_score))
            if e % 5 == 0:
                average = self.Average(5)
                self.agent.SaveWeights(WEIGHTS_NAME)
                print("Average score over 5 games: {0}".format(average))

            if e % TARGET_UPDATE_GAME_INTERVAL == 0 and e is not 0:
                self.agent.UpdateTargetQ()

    def Play(self):
        self.agent.LoadWeights(WEIGHTS_NAME)
        self.render = True
        while True:
            score = self.PlayEpisode(False)
            print("Score {0:.2f}".format(score))
            

if __name__ == "__main__":
    experiment = PongExperiment()
    if TRAIN:
        experiment.Train()
    else:
        experiment.Play()