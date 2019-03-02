# Convolutional Neural Network
Mnist number recognition using a Convolutional neural network.

Convolutional neural network implementation for handwritten number recognition.

# *The structure of the net:*
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 128)       32896     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 256)       131328    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 256)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 5, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6400)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               819328    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,003,658
Trainable params: 1,003,658
Non-trainable params: 0
_________________________________________________________________
```

The model is saved in *'mnist_cnn.h5'* and the current one is trained with **99.98%** accuracy over mnist test dataset.

# Example:
```
sudo pip3 install -r requirements.txt
python3 mnist_cnn.py
```
