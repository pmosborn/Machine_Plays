import os
import sys
import time
from pywinauto import Application
import pyautogui

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

    def send_control(self, btn):
        if not any(btn.lower() == control for control in self.controls):
            return False
        self.dlg.click_input()
        pyautogui.keyDown(btn.lower())
        pyautogui.keyUp(btn.lower())
        return True


if __name__ == '__main__':
    m = machine_player()
    app = m.open_game()
    start = time.time()
    while time.time() - start < 5:
        if not m.send_control("right"):
            print "What"
    m.close_game()