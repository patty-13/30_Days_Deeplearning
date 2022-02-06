# 30_Days_Deeplearning



## Day 1 
  #### Multi Layered Perceptron 
  
  <em>Summary -  It is a class of feedforward Neural Network. Atleast contains 3 layers, Input layer, Hidden Layer, Output Layer for more information refer [here](https://en.wikipedia.org/wiki/Multilayer_perceptron)</em>
   
  The there main type of the layers refer to 
  <ul>
    <em>
      <li>Input Layer</li>
      <li>Hidden Layer</li>
      <li>Output Layer</li>
    </em>
  </ul>
  
  ![MLP model](https://i.imgur.com/McMOhuQ.png?raw=true "Title")
  
  <em><u> Input Layer </u> :-</em>
        This layer refer to the place where we feed attributes to our Neural Network, this is where our features will be feed and propogate through the layers to
        genereate output for our neural network.
        <p>For more information refer this [link](https://stackoverflow.com/questions/32514502/neural-networks-what-does-the-input-layer-consist-of)</p>
        
   <em><u>Hidden Layer :-</u> </em>
        <p>This is the layer that is present between your input layer and output layer, this is where you apply weights to inputs and then directs them through the
        activation functions. Now the question arises what is an activation function, the activation function refers to the function where we apply some kind of 
        transformation to data so that we can make use of it as inputs to next layers and inturn getting answers. </p>
        <p>For more information refer this [link](https://deepai.org/machine-learning-glossary-and-terms/hidden-layer-machine-learning).</p>
        
There are many type of Activation Functions that can be choosed to apply on your data. Refer this [link](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
     
  ![Activation Functions](https://miro.medium.com/max/1400/1*p_hyqAtyI8pbt2kEl6siOQ.png)</p>
    <em><u>Output Layer </u> :- </em>
   <p>This is layer where we get our output for the Neural Network. We can have many outputs depending on the type of problem. For example we can get only a 
  single node as output for a regression problem where as we can get a multiple numeber of outputs for example multiclass classification.</p>
  
  ![Multiclass Classification](https://developers.google.com/machine-learning/crash-course/images/SoftmaxLayer.svg)
  
  For more information refer this 
   [Multiclass_Classification](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax) 

  
  Interview Questions 
  1. Why should the activation function be differentiable ? [Answer](https://www.jeremyjordan.me/neural-networks-activation-functions/#:~:text=An%20ideal%20activation%20function%20is,training%20to%20optimize%20the%20weights.)
  2. When should we use Linear Activation funciton or Non Linear activation function? [Answer](https://stackoverflow.com/questions/9782071/why-must-a-nonlinear-activation-function-be-used-in-a-backpropagation-neural-net)
  3. How does the depth of neural network affects the neural network? [Answer](https://analyticsindiamag.com/depth-in-neural-networks/) 
  4. What is Saturation of Activation Function/ What is vanishing Gradient ? [Answer](https://datascience.stackexchange.com/questions/44213/what-does-it-mean-for-an-activation-function-to-be-saturated-non-saturated), [Alt Answer](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
 
#### Reference
  1. Developers Google 
  2. Stack overflow
  3. Towards data science
  4. Wikipedia
  5. Neural networks: activation functions Jeremy Jordan
  6. Analytics India Mag
## Day 2 
  ### BackPropogation
   
  ## Pratyush add stuff

## Day 3 
  Optimization for Multi Layered Perceptron<br>
  <p>Neural Network uses optimizer to minimize the error in the algorithm and the error is actually computed is by using a         Loss Function which helps in identify how good or bad the model is performing.</p>
1. Loss Function <br> 
    Loss functions measure how far an estimated value is from its true value. We need to know how closely we are estimating our values, this can be done using a loss function that gives us a measure of it. Ideally, we aim to decrease the loss and reach close to our true values.
 
 For more information refer this [link](https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/#:~:text=Loss%20functions%20measure%20how%20far,the%20goal%20to%20be%20met.) 
   
   There are different loss for different task, Below are mentioned the loss according to task
  

1.1 [Regression Loss Function](https://keras.io/api/losses/regression_losses/)
  <ul>
    <em>
      <li>MEAN SQUARED ERROR</li>
      <li>MEAN SQAURED LOGRAITHMIC ERROR LOSS</li>
      <li>MEAN ABSOLUTE ERROR LOSS</li>
  </em>
  </ul>
  
  ![IMAGE](https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/mean-squared-error.PNG)
  
1.2  [Binary Classification Loss Function](https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/)
  <ul>
    <em>
      <li>BINARY CROSS ENTROPY</li>
      <li>HINGE LOSS</li>
  </em>
  </ul>
  
 ![IMAGE](https://chris-said.io/assets/2020_cross_entropy/cross-entropy.png)

  
1.3  [Multi-Class Classification loss function](https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451)
  <ul>
    <em>
      <li>CATEGORICAL CROSS ENTROPY LOSS</li>
      <li>KULLBACK LEIBLER DIVERGENCE LOSS</li>
  </em>
  </ul>
  
 ![IMAGE](http://image.sciencenet.cn/home/201806/13/212841ifa4wdrfgb4awac7.png)
 

2. Optimization
 
   We will talk more about Optimization with specific codes for specific Optimizations also with techniques such as dropout.
  <ul>
    <em>
      <li>GRADIENT DESCENT</li>
      <li>STOCHASTIC GRADIENT DESCENT</li>
      <li>MINI BATCH GRADIENT DESCENT</li>
      <li>MOMENTUM</li>
      <li>NESTEROV ACCELERATED GRADIENT</li>
  </em>
  </ul>
  
   ![optimizer](https://user-images.githubusercontent.com/56751154/151979849-03535af1-0a18-42a8-a177-e4926edd1684.jpg)
      
   <ul>
     <em>
      <li>ADAGRAD</li>
      <li>ADADELTA</li>
      <li>ADAM</li>
      <li>RMSPROP</li>
  </em>
  </ul>
  
  Interview Questions (We will not talk about optimizer questions here)
  
  1. Why dont you use MSE in Logistic Regression ? [Answer](https://www.quora.com/What-are-the-main-reasons-not-to-use-MSE-as-a-cost-function-for-Logistic-Regression#:~:text=The%20main%20reason%20not%20to,the%20function%20to%20optimally%20converge.)
    
  For more information and better explaination refer this [link](https://www.deeplearning.ai/ai-notes/optimization/)

  #### References 
  1. Quora
  2. DeepLearning.ai 
  3. Analystics Vidhya
  4. Towards Data Science  
  
## Day 4 
  Added code for Multi Layered Perceptron using Tensorflow(keras)
  Refer this [link](https://github.com/patty-13/30_Days_Deeplearning/blob/main/Codes/MLP/Multi_Layered_Preceptron.ipynb)
  
## Day 5
  ### Convolutional Neural Network  
  
  CNN (Convolutional Neural Network or ConvNet) is a type of feed-forward artificial network where the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex. 
  
   Refer this [link](http://www.cns.nyu.edu/~david/courses/perception/lecturenotes/what-where/what-where.html)
   
   The best example is to understand how sobel operator detects lines refer this [link](https://en.wikipedia.org/wiki/Sobel_operator) for sobel line detection
   
   We have a kernel that convolves over the image and then gives you a output and on the basis of that output we make decision. In convolutions, we have element wise multiplications and summation over them to get an element for a new matrix. The scope of the multiplications is determined by the size of the filter.
   
   Such distinguished filters are applied to get different features of images for visual tasks in CNN. In convolutions we involve element wise multiplications and additions.
   
   In the sobel opertor we find out that the resultant matrix is smaller than the input matrix to keep them of simillar size we use a term called padding. 
   
   In formulas you can have 
   
   Input matrix - (6,6) 
   Sobel Kernel - (3,3) 
   After convolution we get
   Output Matrix  - Input Matrix *(conv)* Sobel Kernel => (4,4)
   
   The formula for the dimensions of the output matrix is - (n-k+1, n-k+1): n - size of input matrix, k – size of filter matrix (Considering stride = 1)
   
   As we saw that output matrix is smaller than the input matrix so we add padding around the input matrix. Refer this [link](https://deepai.org/machine-learning-glossary-and-terms/padding#:~:text=Padding%20is%20a%20term%20relevant,will%20be%20of%20value%20zero.)
   
   With padding of size p, we will have final matrix of size (n-k+2p+1, n-k+2p+1)
   
   The Example mentioned above works with Grayscale image for getting a more intutive understanding for RGB assume that there are three types of images kept over each other with different pixel values for different colours. It can be assumed more of a 3D cube with 3 different matrix with pixel values of different colour. 
   This [link](https://cs231n.github.io/convolutional-networks/#pool) will help to understand it better.
   
## Day 6 
  Learning Hyper Parameters for CNN
  
  In MLP we learn the weights and baises using backpropogation but in CNN we learn kernel matrices using backpropogation. 
    
 
  [LeNet Architecture](https://www.kaggle.com/blurredmachine/lenet-architecture-a-complete-guide)
  ![Alex Net](https://i0.wp.com/ramok.tech/wp-content/uploads/2017/12/2017-12-31_01h31_40.jpg)
  [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  [VGG16](https://www.quora.com/What-is-the-VGG-neural-network)
  [VGG 16 - ppr](https://arxiv.org/pdf/1409.1556.pdf)
  [VGG 16 Keras Implementation](https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py)
 

## 
  [Resnets Resudial Networks](https://arxiv.org/pdf/1512.03385.pdf)
  [ResNet Code](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)
  
  [Inception Network]( https://arxiv.org/pdf/1512.00567.pdf)
  [Inception Network Code](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py)
  
#### Optimization 

[Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf)
[Adam](https://arxiv.org/pdf/1412.6980.pdf)



















#### About  
This repository is to help people learn Deep Learning in 30 days. 
#### Contributors
[Pratyush Sethi](https://github.com/patty-13) , [Anshuman Singh](https://github.com/Anshuman-37)
