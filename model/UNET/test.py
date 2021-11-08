from UNET import *


model = unet()
# model = unet(2, initial_features=64)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
model.summary()
