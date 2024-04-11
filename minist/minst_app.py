import numpy as np
import torch
from tkinter import *
import tkinter as tk
import win32gui

from PIL import ImageGrab

from minist.minst_train import MinstNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinstNet()
model.load_state_dict(torch.load("minst_model.pth"))
model.eval()


def get_max_possibility(array):
    np.maximum(array, 0, out=array)
    sum = np.sum(array)
    array = np.divide(array, sum)
    return np.max(array)


def predict_digit(img):
    img = img.resize((28, 28))
    # 转灰度
    img = img.convert('L')
    # 转 np 再转 tensor, 并绑定到 cuda 上
    img = np.array(img)
    img = img.reshape(1, 1, 28, 28)
    img = img / 255.0
    img = torch.from_numpy(img).to(torch.float32)

    # 输出为全连接层 [1, 10] 的结果
    output = model(img)
    possibilities = output.data.numpy().copy()
    acc = get_max_possibility(possibilities)
    # 也可以直接使用numpy 最大值的下标
    _, prediction = torch.max(output, dim=1)
    return prediction.numpy()[0], acc


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.title("Minst手写识别")

        # Creating elements
        self.canvas = tk.Canvas(self, width=420, height=420, bg="white", cursor="cross")
        self.label = tk.Label(self, text="结果", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="识别", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="清空画布", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        # todo 目前仅支持 windows app
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        img = ImageGrab.grab(rect)

        digit, acc = predict_digit(img)
        self.label.configure(text="识别结果:{}\n可能性:{:2.1f}%".format(digit, acc*100))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


if __name__ == '__main__':
    # print(model.state_dict())
    app = App()
    tk.mainloop()
