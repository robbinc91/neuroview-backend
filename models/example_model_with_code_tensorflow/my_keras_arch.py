import tensorflow as tf
from tensorflow.keras import layers, Model

class MyResNet3D(Model):
    def __init__(self, filters=16, num_classes=2):
        super(MyResNet3D, self).__init__()
        self.conv1 = layers.Conv3D(filters, 3, padding='same', activation='relu')
        self.pool = layers.GlobalAveragePooling3D()
        self.out = layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return self.out(x)