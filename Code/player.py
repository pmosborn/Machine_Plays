import os
import sys
import time
from pywinauto import Application
import pyautogui
import numpy as np
import skimage
import cv2
from PIL import ImageGrab
from collections import deque
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import random
import json

class machine_player():
    def __init__(self):
        self.pth = "C:\\Program Files (x86)\\In Their Footsteps\\"
        self.game = "AiR.exe"
        self.controls = ['', 'left', 'right', 'space', 'down']
        pyautogui.PAUSE = 0.01
        self.config = 'nothreshold'
        self.actions = 5 # number of valid actions
        self.gamma = 0.99 # decay rate of past observations
        self.observation = 3200. # timesteps to observe before training
        self.explore = 3000000. # frames over which to anneal epsilon
        self.epsilon_final = 0.0001 # final value of epsilon
        self.epsilon_init = 0.1 # starting value of epsilon
        self.replay_memory = 50000 # number of previous transitions to remember
        self.batch = 32 # size of minibatch
        self.frame_per_action = 1
        self.learning_rate = 1e-4
        self.img_rows = 80 
        self.img_cols = 80
        #Convert image into Black and white
        img_channels = 4 #We stack 4 frames

    def open_game(self):
        self.app = Application().start(self.pth+self.game)
        self.dlg = self.app['Made In GameMaker Studio 2']
        self.first_frame = self.get_frame()

    def close_game(self):
        self.app.kill()

    def press_control(self, btn):
        if not any(btn.lower() == control for control in self.controls):
            return False
        pyautogui.keyDown(btn.lower())
        pyautogui.keyUp(btn.lower())
        return True

    def set_control(self, btn):
        if not any(btn.lower() == control for control in self.controls):
            return False
        pyautogui.keyDown(btn.lower())
        return True

    def release_control(self, btn):
        if not any(btn.lower() == control for control in self.controls):
            return False
        pyautogui.keyUp(btn.lower())

    def get_frame(self):
        img = ImageGrab.grab(bbox = (449,169,1472,936))
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        frame = skimage.transform.resize(frame, (80,80))
        frame = skimage.exposure.rescale_intensity(frame, out_range=(0,255))
        return frame.reshape(1, 1, frame.shape[0], frame.shape[1])

    def interact(self, action):
        for i in range(len(action)):
            if action[i] == 1:
                break
        action = self.controls[i]
        reward = 0
        end = False
        prev = self.get_frame()
        self.set_control(action)
        curr = self.get_frame()
        if curr == self.first_frame and not prev == self.first_frame:
            reward = -1
            end = True
        elif action == 'right' and not prev == curr:
            reward = 1
        return curr, reward, end

    def buildmodel(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        model.add(Activation('relu'))
        model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        self.mod = model

    def train_network(self, mode = False):
        self.buildmodel()
        self.open_game()
        self.dq = deque()
        initial = np.zeroes(self.actions)
        initial[0] = 1
        curr, reward, end = self.interact(initial)
        curr = curr/255.0
        state = np.stack((curr, curr, curr, curr), axis = 2)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        if mode:
            self.observe = 999999999
            self.epsilon = self.epsilon_final
            self.mod.load_weights("model.h5")
            adam = Adam(lr=self.learning_rate)
            self.mod.compile(loss='mse',optimizer=adam)    
        else:
            self.observe = self.observation
            self.epsilon = self.epsilon_init
        t = 0
        while (True):
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([self.actions])
            #choose an action epsilon greedy
            if t % self.frame_per_action == 0:
                if random.random() <= self.epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(self.actions)
                    a_t[action_index] = 1
                else:
                    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    a_t[max_Q] = 1

            #We reduced the epsilon gradually
            if self.epsilon > self.epsilon_final and t > self.observe:
                self.epsilon -= (self.epsilon_init - self.epsilon_final) / self.explore

            #run the selected action and observed next state and reward
            x_t1_colored, r_t, terminal = self.interact(a_t)

            x_t1 = x_t1 / 255.0
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            # store the transition in D
            self.dq.append((s_t, action_index, r_t, s_t1, terminal))
            if len(self.dq) > self.replay_memory:
                self.dq.popleft()

            #only train if done observing
            if t > self.observe:
                #sample a minibatch to train on
                minibatch = random.sample(self.dq, self.batch)

                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                targets[range(self.batch), action_t] = reward_t + self.gamma*np.max(Q_sa, axis=1)*np.invert(terminal)

                loss += model.train_on_batch(state_t, targets)

            s_t = s_t1
            t = t + 1

            # save progress every 10000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= self.observe:
                state = "observe"
            elif t > self.observe and t <= self.observe + self.explore:
                state = "explore"
            else:
                state = "train"

if __name__ == '__main__':
    m = machine_player()
    m.train_network()