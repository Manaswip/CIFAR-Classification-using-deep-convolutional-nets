<html>
<body>
<h2> CIFAR-10 dataset classification using deep convolutional networks </h2>
CIFAR 10 dataset contains images belonging to 10 different categories (airplane, automobile, bird, cat, deer, dog, frog, horse, sheep, truck). Images in this dataset do not have backgrounds, therefore any special data preprocessing steps are not required. But dataset is divided into 6 different batches, data from all the batches is retreived and stored in a mat file.
</br> </br>
Code supports different kind of layer types( Full connected layer, 
convoluations layer, max pooling layer, softmax layer) and different activation functions (sigmoid, rectified linear units,
tanh)

Code is built using Theano library so, this code can be run either on CPU or GPU, set GPU to true to run on GPU and set GPU to false to run on CPU

This program incorparates ideas and code from text book on <a href='http://neuralnetworksanddeeplearning.com/index.html'> Neural Networks and Deep learning from Michael Nielsen </a> and <a href='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src'>Michael Nielsen's github</a> 

<h3> Project Layout </h3>
<p>
<b>CIFARProcessingAndLoadingData.py.py:</b> In this program data from 6 different batches is retrieved. Returns a list of training data, testing data, and validation data</br>
<b>CIFARraining_Theano.py:</b> Implementation of deep convolutional networks using theano giving an advantage of running the code either on CPU/GPU. In addition to that this code supports different cost functions and activation functions </br>

import cv2 </br> 
import CIFAR </br> 
from CIFAR import Network </br> 
from CIFAR import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer </br> 
training_data, validation_data, test_data = CIFAR.load_data_shared() </br> 
mini_batch_size = 10 </br> 
net = Network([ </br>
   &emsp;  &emsp;&emsp;&emsp;  ConvPoolLayer(image_shape=(mini_batch_size, 1, 32, 32), filter_shape=(20, 1, 5, 5),  </br>
    &emsp;  &emsp;&emsp;&emsp;                poolsize=(2, 2)), FullyConnectedLayer(n_in=20*14*14, n_out=100), </br>
     &emsp; &emsp;&emsp;&emsp;                SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size) </br> 
net.SGD(training_data, 60, mini_batch_size, 0.1,validation_data, test_data)   </br>

</p>
</body>
</html>
