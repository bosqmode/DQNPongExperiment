import numpy as np
import random
import math
from tensorflow import summary
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import time
from ExperienceReplay import ExperienceReplay


logdir = "logs/log"+str(int(time.time()))
writer = summary.create_file_writer(logdir)

class DQN:
    def __init__(self, input_shape, num_actions, gamma = 0.99, expreplaycap = 20000, batchsize = 64, startepsilon = 1, minepsilon = 0.05, epsilondecay = 0.0001, obsperiod = 100000):
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.gamma = gamma
        self.expreplaycap = expreplaycap
        self.obsperiod = obsperiod
        self.batchsize = batchsize
        self.startepsilon = startepsilon
        self.epsilon = startepsilon
        self.minepsilon = minepsilon
        self.epsilondecay = epsilondecay
        self.opt = keras.optimizers.Adam(learning_rate=0.0001)
        self.initializer = keras.initializers.GlorotNormal(seed=None)
        self.steps = -1

        self.Q = self.CreateModel()
        self.target_Q = self.CreateModel()

        self.lastscore = 0

        self.exp_replay = ExperienceReplay(maxlen=self.expreplaycap)

    def AddSample(self, state, action, reward, next_state, done):
        self.exp_replay.Append((state, action, reward, next_state, done))

    def CreateModel(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Conv2D(64, kernel_size=4, strides=2, activation="relu", kernel_initializer=self.initializer, use_bias = False))
        model.add(Conv2D(64, kernel_size=4, strides=2, activation="relu", kernel_initializer=self.initializer, use_bias = False))
        model.add(Conv2D(256, kernel_size=3, strides=1, activation="relu", kernel_initializer=self.initializer, use_bias = False))
        model.add(Flatten())
        model.add(Dense(units=1024, activation='relu', kernel_initializer=self.initializer))
        model.add(Dense(units=self.num_actions, activation='linear'))
        model.compile(loss='huber', optimizer=self.opt, metrics=['accuracy'])
        return model

    def PredictAction(self, state):
        action = self.Q.predict(state.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        return np.argmax(action)

    
    def EpsilonGreedyAction(self, state):
        self.steps += 1

        if(self.steps>self.obsperiod):
            self.epsilon = max(min(self.epsilon * (1.0-self.epsilondecay) ,self.startepsilon),self.minepsilon)

        if (random.random() < self.epsilon or self.steps < self.obsperiod):
            return random.randint(0, self.num_actions-1)
        else:
            return self.PredictAction(state)

    def UpdateTargetQ(self):
        print("weights copied to target")
        self.target_Q.set_weights(self.Q.get_weights())

    def TrainOnBatch(self, x, y):
        history = self.Q.train_on_batch(x, y)
        return history

    def SaveWeights(self, filename):
        print("weights saved!")
        self.Q.save_weights(filename)

    def LoadWeights(self, filename):
        self.Q.load_weights(filename)
        print("Loaded weights")

    def Train(self):
        if self.exp_replay.Length() < self.expreplaycap:
            return

        batch = self.exp_replay.RandomBatch(self.batchsize)
        batchLen = len(batch)

        batch = list(zip(*batch))
        current_states = np.array(batch[0])
        next_states = np.array(batch[3])

        Qs = self.Q.predict(current_states.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        targetQs = self.target_Q.predict(next_states.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        x = np.zeros((batchLen, self.input_shape[0], self.input_shape[1]))
        y = np.zeros((batchLen, self.num_actions))

        for i in range(batchLen):
            a = batch[1][i]
            reward = batch[2][i]
            targetQ = Qs[i]
            if batch[4][i] >= 1:
                targetQ[a] = reward
            else:
                targetQ[a] = reward + self.gamma * np.max(targetQs[i])

            x[i] = current_states[i]
            y[i] = targetQ

        hist = self.TrainOnBatch(x.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]), y)

        if self.steps % 500 == 0 and self.steps is not 0:
            with writer.as_default():
                summary.scalar("Loss", hist[0], step=self.steps)
                summary.scalar("Accuracy", hist[1], step=self.steps)
                summary.scalar("Epsilon", self.epsilon, step=self.steps)
                summary.scalar("Last episode score", self.lastscore, step=self.steps)
                img = np.reshape(x[0:10], (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
                summary.image("State", img, max_outputs=1, step=self.steps)
                for act_i in range(self.num_actions):
                    summary.scalar("Average Q for action {0}".format(act_i), np.average(Qs[:][act_i]), step=self.steps)
            writer.flush()
    