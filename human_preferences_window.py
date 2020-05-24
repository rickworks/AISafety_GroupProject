import tkinter as tk
import PIL.Image, PIL.ImageTk

class HumanPreferenceWindow():
    """ Window for performing comparisons between two videos about which is better than the other """

    def __init__(self, master, videoWidth = 160, videoHeight = 192, zoom = 2, cartpole=False):
        self._master = master
        self._master.resizable(False, False)
        self._zoom = zoom

        self._video1 = []
        self._frame1 = []
        self._video2 = []
        self._frame2 = []
        self._frameIndex = 0
        self._callback = []

        if cartpole:
            videoWidth=300
            videoHeight=200
            zoom=2

        self._canvas1 = tk.Canvas(self._master, width = videoWidth * zoom, height = videoHeight * zoom, borderwidth = 2, relief="groove")
        self._canvas1.grid(row=0, column=0)

        self._canvas2 = tk.Canvas(self._master, width = videoWidth * zoom, height = videoHeight * zoom, borderwidth = 2, relief="groove")
        self._canvas2.grid(row=0, column=1)

        self._buttonLeft = tk.Button(self._master, text = "Left", command = self._actionLeft, state = tk.DISABLED)
        self._buttonLeft.grid(row=1, column=0, sticky = tk.W + tk.E + tk.N + tk.S)

        self._buttonRight = tk.Button(self._master, text = "Right", command = self._actionRight, state = tk.DISABLED)
        self._buttonRight.grid(row=1, column=1, sticky = tk.W + tk.E + tk.N + tk.S)

        self._buttonSame = tk.Button(self._master, text = "About same", command = self._actionSame, state = tk.DISABLED)
        self._buttonSame.grid(row=2, column=0, columnspan=2, sticky = tk.W + tk.E + tk.N + tk.S)

        self._buttonNeither = tk.Button(self._master, text = "Bad comparison", command = self._actionNeither, state = tk.DISABLED)
        self._buttonNeither.grid(row=3, column=0, columnspan=2, sticky = tk.W + tk.E + tk.N + tk.S)

        self._updateScheduled = False
    ##

    def compare(self, video1, video2, callback):
        """ Compare the two given videos and return a label reporting which is best """
        if self._video1:
            raise("Already comparing a pair of videos!")
        ##
        self._video1 = video1
        self._video2 = video2
        self._frameIndex = 0
        self._callback = callback
        self._buttonLeft.config(state=tk.NORMAL)
        self._buttonRight.config(state=tk.NORMAL)
        self._buttonSame.config(state=tk.NORMAL)
        self._buttonNeither.config(state=tk.NORMAL)
        if not self._updateScheduled:
            self._updateLoop()
        ##
    ##

    def _updateLoop(self):
        # Update function to be called every 1/60 s when video is playing.
        if not self._video1:
            self._updateScheduled = False
            return
        ##
        img = PIL.Image.fromarray(self._video1[self._frameIndex])
        if self._zoom > 1:
            img = img.resize((img.width * self._zoom, img.height * self._zoom), PIL.Image.BILINEAR)
        ##
        self._frame1 = PIL.ImageTk.PhotoImage(image = img)
        self._canvas1.create_image(0, 0, image=self._frame1, anchor=tk.NW)
        img = PIL.Image.fromarray(self._video2[self._frameIndex])
        if self._zoom > 1:
            img = img.resize((img.width * self._zoom, img.height * self._zoom), PIL.Image.BILINEAR)
        ##
        self._frame2 = PIL.ImageTk.PhotoImage(image = img)
        self._canvas2.create_image(0, 0, image=self._frame2, anchor=tk.NW)
        self._frameIndex += 1
        if (self._frameIndex >= len(self._video1)):
            self._frameIndex = 0
        ##
        self._updateScheduled = True
        self._master.after(15, self._updateLoop)
    ##

    def _stop(self):
        self._video1 = []
        self._frame1 = []
        self._video2 = []
        self._frame2 = []
        self._buttonLeft.config(state=tk.DISABLED)
        self._buttonRight.config(state=tk.DISABLED)
        self._buttonSame.config(state=tk.DISABLED)
        self._buttonNeither.config(state=tk.DISABLED)

    def _actionLeft(self):
        self._stop()
        self._callback("1>2")
    ##

    def _actionRight(self):
        self._stop()
        self._callback("1<2")
    ##

    def _actionSame(self):
        self._stop()
        self._callback("1=2")
    ##

    def _actionNeither(self):
        self._stop()
        self._callback("n/a")
    ##
##

