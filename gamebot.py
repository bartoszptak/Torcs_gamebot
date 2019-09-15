import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

import tensorflow as tf
tf.random.set_random_seed(2137)
np.random.seed(2137)
random.seed(2137)

class Torcs:
    def __init__(self, model, train):
        self.INITIAL_EPSILON = 0.1
        self.FINAL_EPSILON = 0.0001
        self.EXPLORE = 100000
        self.GAMMA = 0.99
        self.batch_size = 16

        self.train = train
        self.model = model
        self.deque = deque(maxlen=5000)

        if train:
            self.epsilon = self.INITIAL_EPSILON
        else:
            self.epsilon = self.FINAL_EPSILON

    def preprocessing(self, data):
        img = data.img
        Xaxis = data.speedX
        Yaxis = data.speedY

        img = img * 1./255.
        Xaxis = (Xaxis+1)/2.
        Yaxis = (Yaxis+1)/2.

        return np.array([img]), np.array([Xaxis, Yaxis])

    def make_action(self, img, speed):
        if random.random() < self.epsilon:
            steer, acc = np.tanh(np.random.randn(2))
        else:
            steer, acc = self.model.predict([img, speed])
            steer, acc = steer[0][0], acc[0][0]

        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON -
                             self.FINAL_EPSILON) / self.EXPLORE

        if acc>=0:
            gas = acc
            brake = 0.
        else:
            gas = 0.
            brake = acc * (-1)

        return np.array([steer,gas,brake])

    def make_buffer(self, img, speed, next_img, next_speed, reward, terminal):
        self.deque.append((img, speed, next_img, next_speed, reward, terminal))

    def make_train(self):
        if len(self.deque) < self.batch_size:#*50:
            return 0, 'WAIT'

        minibatch = random.sample(self.deque, self.batch_size)
        
        img, speed, next_img, next_speed, reward, terminal = zip(*minibatch)

        img, next_img = np.concatenate(img), np.concatenate(next_img)
        speed, next_speed = np.concatenate(speed), np.concatenate(next_speed)

        targets = self.model.predict([img, speed])
        Q = self.model.predict([next_img, next_speed])

        print(targets, Q)