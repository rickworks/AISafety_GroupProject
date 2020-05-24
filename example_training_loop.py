from human_preferences_window import HumanPreferenceWindow
from pong_tragetory_generator import PongTragetoryGenerator
import tkinter as tk
import pickle
import numpy as np
import os.path

class ExampleTrainingLoopApp:
    """ Example training loop """

    _num_initial_tragetories = 10
    _num_initial_comparisons = 10
    _num_epochs = 3

    def __init__(self, master):
        self._master = master
        self._master.title("Preference indicator Epoch 1/" + str(self._num_epochs))
        self._window = HumanPreferenceWindow(master)

        self._epoch = 0
        self._num_tragetories = 0
        self._comparisons = []

        self._updateTitle()
        self._master.after_idle(self._trainAlgorithmAndGeneratePredictors)
    ##

    def _trainAlgorithmAndGeneratePredictors(self):
        # On first epoch, we need to seed with some form of agent.
        # TODO: Random agent should only be first few epochs
        generator = PongTragetoryGenerator(RandomBaselineAgent(), RandomBaselineAgent())
        for idx in range(self._num_initial_tragetories):
            self._saveTragetory(idx, generator.build())
        ##
        self._num_tragetories = self._num_initial_tragetories

        # On subsequent epochs, we train one using reward predictor.
        # TODO: Need to add something here
        # Probably ought to save here too
        if self._epoch == self._num_epochs:
            self._master.destroy()
            return
        #

        self._master.after_idle(self._generateComparisons)
    ##

    def _trainRewardPredictor(self):
        # Reward predictor training
        # TODO: Need to add something in here.
        self._master.after_idle(self._nextEpoch)
    #


    def _generateComparisons(self):
        # Generate comparisons with human input
        self._comparisons = []
        self._compareCallback(0, 0, "n/a")
    #

    def _compareCallback(self, idx1, idx2, label):
        # Callback for human selection in GUI 
        if label != "n/a":
            self._comparisons.append((idx1, idx2, label))
        ##

        if len(self._comparisons) == self._num_initial_comparisons:
            self._master.after_idle(self._trainRewardPredictor)
        else:
            idx1 = np.random.randint(0, self._num_tragetories)
            idx2 = np.random.randint(0, self._num_tragetories - 1)
            if idx2 >= idx1:
                idx2 += 1
            ##
            tragetory1 = self._loadTragetory(idx1)
            tragetory2 = self._loadTragetory(idx2)
            self._updateTitle()
            self._window.compare(tragetory1, tragetory2, lambda label: self._compareCallback(idx1, idx2, label))
        ##
    ##

    def _saveTragetory(self, idx, tragetory):
        # Save tragetory of given index
        filename = self._getTragetoryFilename(idx)
        with open(filename, 'wb') as output:
            pickle.dump(tragetory, output)
        ##
    ##

    def _updateTitle(self):
        # Update the title to match current info
        self._master.title("Preference indicator"
            + " Epoch " + str(min(self._epoch + 1, self._num_epochs)) + "/" + str(self._num_epochs)
            + " Comparison " + str(min(len(self._comparisons) + 1, self._num_initial_comparisons)) + "/" + str(self._num_initial_comparisons))
    ##

    def _nextEpoch(self):
        # Called at the start of each epoch
        
        self._epoch += 1
        self._updateTitle()
        self._master.after_idle(self._trainAlgorithmAndGeneratePredictors)
    ##

    def _loadTragetory(self, idx):
        # Load tragetory of given index
        filename = self._getTragetoryFilename(idx)
        with open(filename, 'rb') as input:
            return pickle.load(input)
        ##
    ##

    def _getTragetoryFilename(self, idx):
        # Get the filename for a given tragetory
        if not os.path.exists("tragetories"):
            os.mkdir("tragetories")
        ##
        return "tragetories/" + str(idx) + ".pkl"
    #

#

class RandomBaselineAgent:
    """ an agent to play pong that always just does random actions """

    def showdown_step(self,
                      observation,
                      reward):
        """
        args:
        observation = matrix of pixels from pong (ignored)
        reward = float (ignored)
        returns:
        action = binary array of length 8
        """
        return np.random.randint(0, 2, 8)
##

if __name__ == "__main__":
    root = tk.Tk()
    app = ExampleTrainingLoopApp(root)
    root.mainloop()
##
