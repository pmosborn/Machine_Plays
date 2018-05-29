import os
import sys
import time
from pywinauto import Application
import pyautogui
import numpy as np
import skimage
import cv2
from PIL import ImageGrab

class machine_player():
    def __init__(self):
        self.pth = "C:\\Program Files (x86)\\In Their Footsteps\\"
        self.game = "AiR.exe"
        self.controls = ['left', 'right', 'space', 'down']
        pyautogui.PAUSE = 0.01

    def open_game(self):
        self.app = Application().start(self.pth+self.game)
        self.dlg = self.app['Made In GameMaker Studio 2']

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
        frame = frame.reshape(1, 1, frame.shape[0], frame.shape[1])


if __name__ == '__main__':
    m = machine_player()
    app = m.open_game()
    m.dlg.click_input()
    start = time.time()
    while time.time() - start < 5:
        m.set_control("right")
    m.release_control("right")
    m.press_control('space')
    start = time.time()
    while time.time() - start < 3:
        continue
    m.close_game()