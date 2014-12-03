"""This module contains function to get the necessary datasets and mlpython
problem for the project."""
from __future__ import print_function
import glob
import re
from scipy.misc import imread
from numpy import random
import numpy as np
from mlpython.mlproblems.classification import ClassificationProblem


def load_data(folder):
    """Load all the images files into memory. It puts them in a list of tuple containing the
    filename and the images. The images are in the same format as the file (rgb)."""
    folder.rstrip("/ ")
    filenames = glob.glob(folder + "/*.png")
    return ((imread(filename, flatten=True), filename) for filename in filenames)


def create_label(filename):
    """Uses a regular expression to find which objects corresponds to this filename and what is
    its orientation."""
    found = re.search(r'(\d{0,4}__\d{1,4})', filename).group(0).split('_')
    return int(found[0]), int(found[-1])


def create_labels(data):
    """Returns a generator which convert the filenames into (obj, orientation) tuple"""
    return ((image, create_label(filename)) for image, filename in data)


def divide_problem(data, seed=1234):
    random.seed(seed)
    indexes = list(range(0, 100))
    random.shuffle(indexes)
    train_indexes = indexes[0:30]
    train, test, valid, video = [], [], [], []
    for image, (sequence, angle) in data:
        if sequence in train_indexes:
            if angle in [0, 90, 180, 270]:
                train.append((image, sequence))
            elif angle in [45, 135, 225, 315]:
                valid.append((image, sequence))
            else:
                test.append((image, sequence))
        else:
            video.append(image)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(valid)
    return train, valid, test, video


def create_problem(data, seed=1234):
    train, valid, test, video_sequences = divide_problem(data)
    targets = set((i[1] for i in train))
    meta = {'targets':targets, 'input_size':len(train[0][0])}
    trainset = VideoClassification(data=train, video=video_sequences, metadata=meta)
    validset = VideoClassification(data=valid, video=video_sequences, metadata=meta)
    testset = VideoClassification(data=test, video=video_sequences, metadata=meta)
    return trainset, validset, testset

def get_classification_problem(img_folder):
    data = load_data(img_folder)
    data = create_labels(data)
    return create_problem(data, seed=1234)

class VideoClassification(ClassificationProblem):
    def __init__(self, data=None, video=None, metadata=None, seed=1234, call_setup=True):
        if metadata is None:
            metadata = {}
        ClassificationProblem.__init__(self, data=data, metadata=metadata, call_setup=call_setup)
        self.video = np.array(video)

    def getConsecutivesFrames(self, n=1):
        indices = random.randint(0, self.video.shape[0], (n, ))
        indices_next = indices + 1
        return self.video[indices, :], self.video[indices_next, :]



    def getNonConsecutivesFrames(self, n=1):
        indices = random.randint(0, self.video.shape[0], (n, ))
        indices_next = random.randint(0, self.video.shape[0], (n, ))
        return self.video[indices, :], self.video[indices_next, :]


if __name__ == "__main__":
    import time
    path = "../images/"
    data = load_data(path)
    #data = list(create_labels(data))
    data = create_labels(data)
    #t = time.time()
    train, valid, test = create_problem(data, seed=1234)
    #print(time.time() - t)
