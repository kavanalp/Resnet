# Resnet



2.1  Introduction for Resnet
Previously, it was believed that the more layers in a neural network, the better the performance. As a result, a four-layer network will certainly perform better than a three-layer one. Further research disputes this. Deeper networks may decrease accuracy even during training.
Reset comes into play to ensure that the deep layers do not hinder the whole network's performance.
In this way, information can be saved from the previous layer to the next layer. If layer N loses some information, we connect layer N-1 to layer N+1 to hold that information in our network.
2 kinds of the block connect N-1 to N+1:

Identity block: This block is considered when the shapes N-1 and N+1 
are identical. It just applies an activation function (which is usually Relu)to N-1 and then adds up to N+1

Convolution block: This block is considered when the shapes N-1 and N+1 
are different. So the block would be convolution with the same 1*1 kernel and same stride and padding as the Nth.

2.2 Dataset
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."
Zalando seeks to replace the original MNIST dataset
 
2.3 Purpose
The result of Resnet and our own architecture will be combined
 
2.4 Preprocessing
The dataset has 784 columns so as an image we reshape it to a 28 * 28 array. Then we divide their value by 255 for decreasing the calculation cost. We chose 25 % of the data as our validation. Finally, we have another dataset as our test set with 10000 columns. We set our batch size to 100 and then build our data loader with PyTorch.
 
2.5 Model
We use Resnet and a regular CNN for this project
 
2.5.1 ResNet
	We use ResNet 18 architecture which has 18 hidden layers. the default number of input channels in the ResNet is 3, but our images have 1 channel(gray and white). So we have to change 3 to 1. The number of classes in our dataset is 10. The default is 1000 So we change the Fully connected to 10. And we chose our batch size to be 20 and our learning rate to 0.001 with a decay value of (gamma) 0.1 in 5 steps. It means in every 5 epochs the learning rate decreases to 0.1 of its value.
And with the name of GOD, we start training. 
It took 25 minutes in Colab GPU. and 82 minutes with my local CPU. 
2.5.1 CNN
	I just add 2 simple convolutional layers with 3 fully connected layers as Linear blocks .of course for best comparison I should have taken the exact 18 layers of ResNet with that skipping blocks as our assumption but believe it takes too much to generate and training would definitely get lower accuracy. So, I just used a regular Network to just evaluate the Resnet18 for this kind of data. It took only 3 minutes, and I was amazed at how accurate it was.
.   Make a model class (FashionCNN in our case)
    . It inherit nn.Module class that is a super class for all the neural networks in Pytorch.
. Our Neural Net has following layers:
    . Two Sequential layers each consists of following layers-
        . Convolution layer that has kernel size of 3 . 3, padding = 1 (zero_padding) in 1st layer and padding = 0 in second one. Stride of 1 in both layer.
        . Batch Normalization layer.
        . Acitvation function: ReLU.
        . Max Pooling layer with kernel size of 2 * 2 and stride 2.
     . Flatten out the output for dense layer(a.k.a. fully connected layer).
     . 3 Fully connected layer  with different in/out features.
     . 1 Dropout layer that has class probability p = 0.25.
  
     . All the functionaltiy is given in forward method that defines the forward pass of CNN.
     . Our input image is changing in a following way:
        . First Convulation layer : input: 28 \* 28 \* 3, output: 28 \* 28 \* 32
        . First Max Pooling layer : input: 28 \* 28 \* 32, output: 14 \* 14 \* 32
        . Second Conv layer : input : 14 \* 14 \* 32, output: 12 \* 12 \* 64
        . Second Max Pooling layer : 12 \* 12 \* 64, output:  6 \* 6 \* 64
    . Final fully connected layer has 10 output features for 10 types of clothes.
 
 

 
You can see learning curves in each iteration as we chose adam for our optimization the oscillation is too much for Loss so we can try other optimations for faster convergence but this is out of the concept of this project
 
 
2.6 Evaluation
 As you can see in the figures, Precision-Recall and F1 Score are our metrics for these two models.
 

This is a grouped bar plot and the X axis is our out put class and Y axis is the value of each class in Precisoin just in Dress class our cnn works better but in other classes ResNet wons the competition.
 

And The recall is the rate of true positive divided by the sum of true positive and false negative.
exactly like recall, ResNet works better than our CNN
 

And The f1-score is a harmonic average of precision and recall.
As with recall, ResNet works far better than CNN
 
Final word
Fashion is not a very complex dataset, and a shallow network get an acceptable result. Although ResNet works perfectly in all our metrics Due to the KISS rule we better to chose the simpler model except in competitions.
We chose CNN if time and calculation cost matter, otherwise we chose ResNet Architecture.



