"""
Simple aritificial neural network for multi-clssification

id: Sequential id
huml: Length of Humerus (mm)
humw: Diameter of Humerus (mm)
ulnal: Length of Ulna (mm)
ulnaw: Diameter of Ulna (mm)
feml: Length of Femur (mm)
femw: Diameter of Femur (mm)
tibl: Length of Tibiotarsus (mm)
tibw: Diameter of Tibiotarsus (mm)
tarl: Length of Tarsometatarsus (mm)
tarw: Diameter of Tarsometatarsus (mm)
type: Ecological Group
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def bird_dataset(path="datasets/bird.csv"):
    """Loading iris dataset and split training/testing data"""
    iris_data = pd.read_csv(path)

    iris_x = iris_data[VARIABLES].values
    iris_y = np.array([0 if y=='Iris-setosa' else 1 for y in iris_data['species']])

    return train_test_split(iris_x, iris_y, test_size=TEST_PROPORTION)
