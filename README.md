# 30_Days_Deeplearning


## Day0
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
  After Forward Propagation and comparision with the results, to update the parameters values propagates in the backward direction, simply to update the parameters the backward movement is called BackPropagation. So it is an algorithm used to   calculate derivatives quickely.<br>
 ### Working of BackPropagation
 So, the main three layers are:
 1. input layer
 2. hidden layer
 3. output layer<br>
 ![image](https://user-images.githubusercontent.com/56751154/155643429-4d035c79-9da3-4767-a31c-6c9aace1aafa.png)<br>
 The image summarizes the functioning of the backpropagation approach.<br>
 Forward propagation:
 1. Input layer receives inputs or x.
 2. input is modeled using weights w.
 3. Each hidden layer calculates z = w.x + b, then sends it through acitvation. A(Z).
 4. In output layer the difference between acutal value and the value obtained is called error.
 Back Propagation
 5. Go back to the hidden layers and adjust the weights so that this error is reduced in future runs.<br>
 So basically this what happens in equation form
 ![image](https://user-images.githubusercontent.com/56751154/155645025-a5ee2475-331c-4e56-98f3-51383836831c.png)
### Forward Prop mathematical equations.
![image](https://user-images.githubusercontent.com/56751154/155645120-c9f4f4c1-de7d-477e-a40a-2565089c658e.png)
### Backward Prop mathematical equations.
  ![image](https://user-images.githubusercontent.com/56751154/155645158-1f114024-374f-4489-8346-68df69ccaf2a.png)<br>
#### Reference<br>
https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60<br>
https://www.mygreatlearning.com/blog/backpropagation-algorithm/
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
  STEP BY STEP IMPLEMENTATION OF MULTILAYERD PRECEPTRON<br>
  The code encapsulates the knowledge gained from the content learned in day 1,2,3.<br>
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
   
   Interview Questions 
   ### some of the common questions realtead to CNN.
1. Why is CNN preferred over ANN for image data?
2. What is the importance of the RELU activation function in CNN?
3. Explain the use of the pooling layer in CNN.
4. Explain the difference between valid padding and the same padding in CNN.
5. Explain the role of a fully connected (FC) layer in CNN.
6. What is the importance of parameter sharing 
7. Explain the different types of Pooling. 
8. What is the use of the convolution layer in CNN?
9. What are the advantages of using CNN over DNN?
10. How would you visualise features of CNN in an image classification task?
11. What do you understand by shared weights in CNN?
12. Explain the process of flattening?
13. Can CNN be used to perform Dimensionality Reduction? If yes, how?
14. Define the term sparsity of connections in CNN.
15. List the hyperparameters of a pooling layer in CNN
   
   References
   1. Analytics vidhaya
   2. blog insiad
   3. wikipedia
   4. deepai
   
## Day 6 
  Learning Hyper Parameters for CNN
  
  In MLP we learn the weights and baises using backpropogation but in CNN we learn kernel matrices using backpropogation. 
  At each convolution layer we will have multiple kernels to learn different features of the images. For each kernel we will have a 2D array output, multiple kernels result in multiple output arrays (padded to get input array size), so at the layer we will get an output array of size n x n x m, where m is number of kernels (m is a hyper parameter). 
  
  For every output element obtain from the filters we appy activation function. 
  
  To put it simply we pad, convolve and activate to transform input array in convolution layer. 
  For MLP we train weights, for CNN we learn kernel. 
  
  Then we have a concept of pooling refer this [link](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) to get a better understanding. 
  
  Main point of pooling is to introduce concept of Local invariance, Scale invariance and Rotational invariance.
  But, sometimes invariance is not desirable. We can also have a goal of equivariance, where a small change in input array should also reciprocate a small change in output (invariance: a change in input image does not show changes in output array)
   
   Refer to this [link](https://www.slideshare.net/kuwajima/cnnbp) to get a better understanding for how to optimize CNN models. 
   
  ## Day 7
  IMPLEMENTATION OF CNN
  
  1. Step by Step implementation of CNN. (NOTE: I WILL BE ADDING IT TODAY)
  2. keras implementation of CNN.  (NOTE: I WILL BE ADDING IT TODAY)
  
  ## Day-8
  FAMOUS CNN ARCHITECTURES
  1. LeNet-5 (1998)
     [LeNet Architecture](https://www.kaggle.com/blurredmachine/lenet-architecture-a-complete-guide)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155654094-fd2f7854-e587-4136-a57c-9d4c539d1fd1.png)<br>
     LeNet-5 is one of the simplest and the first architecture. This architecutre has become the standard 'template': stacking<br>
     convolutions with activation funciton, and pooling layers, and ending the network with one or more fully-connected layers.
     It has:
     <ul>
        <li>2 convolutional layer</li>
        <li>3 fully connected layer</li>
        <li>The average-pooling layer</li>
        <li>trainable weights</li>
     </ul>
    Paper : http://yann.lecun.com/exdb/publis/index.html#lecun-98
    Authors: Yann leCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner
 

  2. AlexNet (2012)
     [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155654642-29f84d60-399d-47e7-9efc-c578d412055a.png)<br>
     AlexNet is the improved version of LeNet-5, with 60M parameters. This model was the first to use ReLU as activation function.      and Dropouts. So basically AlexNet just stacked a few more layers on LeNet-5.
     it has:
     <ul>
        <li>5 convolutional layer</li>
        <li>3 fully connected layer</li>
        <li>The max-pooling layer</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
    Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton. University of Toronto, Canada.
  
  ## Day-9
  FANOUS CNN ARCHITECTURE
  
  3. VGG-16(2014)
     [VGG 16 - ppr](https://arxiv.org/pdf/1409.1556.pdf)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155656375-630e2bca-dfd9-478f-9cf3-c5ad679622f3.png)<br>
     The main point here is that CNN are getting deeper and deeper. This is because the most straightforward way is to increase        the number of layer or size. Including 13 convolutional and 3 fully connected layer, ReLU is also used from AlexNet.So In          this:<br><br>
     IMPLEMENTATION OF VGG-16: [VGG 16 Keras Implementation](https://github.com/fchollet/deep-learning-models/blob/master/vgg16.py)<br>
     <ul>
        <li>13 convolutional layer</li>
        <li>3 fully connected layer</li>
        <li>The max-pooling layer</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1409.1556
    Authors: Karen Simonyan, Andrew Zisserman. University of Oxford, UK.
    
    
  4. Inception-v1(2014)
     [Inception Networks](https://arxiv.org/pdf/1512.00567.pdf)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155657671-2147ce4f-775c-43b8-befe-83fa58cc150d.png)<br>
     This is a 22-layer architecture with 5M parameters. The Network in Network is implemented via Inception modules. The design        of the module is a product of research on approximating sparse structure. So in this the networks are build using modules/blocks instead of stacking convolutional layers.<br><br>
     IMPLEMENTATION OF INCEPTION: [Inception Network Code](https://github.com/keras-team/keras/blob/master/keras/applications/inception_v3.py)<br>
     <ul>
        <li>parallel towers of convolutional with different filters, followed by concentration, captures different features 1x1,            3x3 and 5x5</li>
        <li>1x1 convolutions are used for dimensionality reduction to remove bottlenects</li>
        <li>Two auxiliary classifiers to encourage discrimination in the lower stages of the classifier, to increase the gradient signal that gets propagated back, and provide additional regualriation </li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1409.4842
    Authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. Google, University of Michigan, University of North Carolina.

    
  5. Inception-v3 (2015)
     [Inception_Network](https://arxiv.org/abs/1512.00567)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155659519-e14677fa-6357-4411-8a7f-68282ebac9b2.png)<br>
      Inception-v3 is a successor to Inception-v1, with 24M parameters. This model is among first to use batch normalization.
      <br><br> IMPLEMENTATION CODE: [Inception-v3](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py)<br>
      <ul>
        <li>Factorising n×n convolutions into asymmetric convolutions: 1×n and n×1 convolutions</li>
        <li>Factorise 5×5 convolution to two 3×3 convolution operations</li>
        <li>Replace 7×7 with a series of 3×3 convolutions</li>
        <li>max pooling</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1512.00567
    Authors: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. Google, University College London
  ## Day-10
  FAMOUS CNN ARCHITECUTRE
  
  6. ResNet-50 (2015)
     [Resnets Resudial Networks](https://arxiv.org/pdf/1512.03385.pdf)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155660472-b2d23ff8-96e0-42fd-8a07-e6c3927ea31c.png)<br>
     ![image](https://user-images.githubusercontent.com/56751154/155660601-794875ad-5faa-4fb5-8d44-7c9cdfc30dd9.png)<br>
      From the past few CNNs, we have seen nothing but an increasing number of layers in the design and achieving better                 performance. But “with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades       rapidly.” The folks from Microsoft Research addressed this problem with ResNet — using skip connections (a.k.a. shortcut           connections, residuals) while building deeper models.
      ResNet is one of the early adopters of batch normalisation (the batch norm paper authored by Ioffe and Szegedy was submitted       to ICML in 2015). Shown above is ResNet-50, with 26M parameters.<br><br>
     IMPLEMENTATION CODE: [ResNet-50](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)<br>
      <ul>
         <li>Popularised skip connections (they weren’t the first to use skip connections)</li>
         <li>Designing even deeper CNNs (up to 152 layers) without compromising the model’s generalisation power</li>
         <li>Among the first to use batch normalisation</li>
         <li>max pooling</li>
         <li>trainable weights</li>
       </ul>
     Paper : https://arxiv.org/pdf/1512.03385.pdf<br>
     Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Microsoft.
    
    
  8. Xception (2016)
     [Xception-ppr](https://arxiv.org/abs/1610.02357)
    ![image](https://user-images.githubusercontent.com/56751154/155662383-327a351a-43e5-4abb-835d-2489cb90e2d1.png)<br>
      Xception is an adaptation from Inception, where the Inception modules have been replaced with depthwise separable                 convolutions. It has also roughly the same number of parameters as Inception-v1 (23M). Introduced CNN based entirely on           depthwise separable convolution layers.<br><br>
      IMPLEMENTATION OF CODE: [Xception-Code](https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py)<br>
      <ul>
        <li>Firstly, cross-channel (or cross-feature map) correlations are captured by 1×1 convolutions</li>
        <li>Consequently, spatial correlations within each channel are captured via the regular 3×3 or 5×5 convolutions</li>
        <li>Taking this idea to an extreme means performing 1×1 to every channel, then performing a 3×3 to each output. This is identical to replacing the Inception module with depthwise separable convolutions.</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1610.02357
    Authors: François Chollet. Google
  
  ## Day-11
  FAMOUS CNN ARCHITECTURE conti..
  
  8. Inception v-4
     [Inception v-4 ppr](https://arxiv.org/abs/1602.07261)<br><br>
     ![image](https://user-images.githubusercontent.com/56751154/155664328-d94411df-08ed-41f3-bbe1-dbf538d7d26b.png)<br>
     The folks from Google strike again with Inception-v4, 43M parameters. Again, this is an improvement from Inception-v3. The        main difference is the Stem group and some minor changes in the Inception-C module. The authors also “made uniform choices        for the Inception blocks for each grid size.” They also mentioned that having “residual connections leads to dramatically          improved training speed.”<br><br>
     IMPLEMENTATION CODE: [in-v-4 code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)
     <br>
     <ul>
        <li>Change in Stem module</li>
        <li>Adding more Inception modules.</li>
        <li>Uniform choices of Inception-v3 modules, meaning using the same number of filters for every module.</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1602.07261
    Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Google

  9. Inception-ResNets(2016)
     [I-ResNets ppr](https://arxiv.org/abs/1602.07261)
     ![image](https://user-images.githubusercontent.com/56751154/155664723-ddf7eba1-9d45-493e-846a-2a039517d4f0.png)<br>
     In the same paper as Inception-v4, the same authors also introduced Inception-ResNets — a family of Inception-ResNet-v1 and Inception-ResNet-v2. The latter member of the family has 56M parameters.<br><br>
     IMPLEMENTATION OF CODE: [I-ResNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)<br>
     <ul>
        <li>Converting Inception modules to Residual Inception blocks</li>
        <li>Adding more Inception modules.</li>
        <li>Adding a new type of Inception module (Inception-A) after the Stem module.</li>
        <li>trainable weights</li>
     </ul>
    Paper : https://arxiv.org/abs/1602.07261
    Authors: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Google

  11. ResNeXt-50 (2017)
      [ResNeXt-50](https://arxiv.org/abs/1611.05431)<br><br>
      ![image](https://user-images.githubusercontent.com/56751154/155665075-6d56551e-358f-4c7f-9852-b2e348c20b7d.png)<br>
      If you’re thinking about ResNets, yes, they are related. ResNeXt-50 has 25M parameters (ResNet-50 has 25.5M). What’s different about ResNeXts is the adding of parallel towers/branches/paths within each module, as seen above indicated by ‘total 32 towers. Scaling up the number of parallel towers (“cardinality”) within a module (well I mean this has already been explored by the Inception network, except that these towers are added here)<br><br>
    Paper : https://arxiv.org/abs/1611.05431<br>
    Authors: Min Lin, Qiang Chen, Shuicheng Yan. National University of Singapore<br>
#### Important Links
1. [Algorithm unrolling](https://arxiv.org/pdf/1912.10557.pdf)<br>
2. [LeNet Architecture](https://www.kaggle.com/blurredmachine/lenet-architecture-a-complete-guide)<br>
3. [VGG16](https://www.quora.com/What-is-the-VGG-neural-network)<br>
4. [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)<br>
5. [Training machines how to read](https://proceedings.neurips.cc/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf)<br>
6. [Dropouts](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)<br><br>

#### Optimization 

1. [Batch Normalization](https://arxiv.org/pdf/1502.03167v3.pdf)
2. [Adam](https://arxiv.org/pdf/1412.6980.pdf)

#### Refrences
1. Medium
2. wikipedia

      
  
  ## Day-12
  
  ### RNN
  RNN research paper -  Here - [1](https://cseweb.ucsd.edu/~gary/258/jordan-tr.pdf),[2](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)<br>
  LSTM - [blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) , [Research Paper]()<br>
  GRU - [blog](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) , [Research Paper]    (https://arxiv.org/pdf/1409.1259.pdf)<br>
 Progress in RNN - [Slides](https://www.slideshare.net/hytae/recent-progress-in-rnn-and-nlp-63762080)<br>
 Auto Encoders - [Notes](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)<br>
 
  [Resnets Resudial Networks](https://arxiv.org/pdf/1512.03385.pdf)<br>
  [ResNet Code](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)<br>
  
  Word2VEc - [Blog](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)
  
  Deep Visual-Semantic Alignments - [Paper](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf) - CNN gives context vector which is used by the RNN 
  
  Sequence to Sequnec Learning - [Paper](https://arxiv.org/pdf/1409.3215.pdf) - Simple LSTM 
  
  Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation - [Paper](https://arxiv.org/pdf/1406.1078.pdf) Lstm with 2 inputs 
  
  [Seq2Seq CODE](https://github.com/farizrahman4u/seq2seq/blob/master/seq2seq/models.py)
  
  Applications
  
  Translate - [Blog](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)
  
  Email Auto Reply - [Blog](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)
  
  Code Errors - [Blog](https://medium.com/@martin.monperrus/sequence-to-sequence-learning-program-repair-e39dc5c0119b)
  
  Image Captioning - [Blog](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)
  
  
  ## English to French
  
  Data Set - [Website](http://www.manythings.org/anki/)
  
  Character Level Model - [Blog](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)(Code)
  
  ## Day-13
  
  Attention Based Models - [Paper1](https://arxiv.org/pdf/1409.0473.pdf) [Paper2](https://arxiv.org/pdf/1508.04025.pdf) , [Code](https://github.com/facebookresearch/fairseq)
  
  Transformers - [Blog](http://jalammar.github.io/illustrated-transformer/)
  
  Decoding time steps for transformers - 
  ![1](http://jalammar.github.io/images/t/transformer_decoding_1.gif) | [here for second gif](http://jalammar.github.io/images/t/transformer_decoding_2.gif)
  
  ### Attention is all you need - [Paper](https://arxiv.org/abs/1706.03762)
  Bert - [Blog](http://jalammar.github.io/illustrated-bert/)  
  ### IMPLEMENTATION OF BERT FOR TRANSLATION
  BERT implementation code [github repo](https://github.com/patty-13/Neural-Machine-translation-bert-)
  
  Object Detection - Yolov3 - [Main Blog](https://pjreddie.com/darknet/yolo/) , [Working](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)
  
  
  GANS - [Some theory](https://deepgenerativemodels.github.io/) , [Applications](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
  
  
  
  ## Day-14
  ## Day-15
  
  ## Day-16
  
  ## Day-17
  ## Day-18
  
  ## Day-19
  ## Day-20
  ## Day-21
  ## Day-22
  ## Day-23
  
  
  ## Day-24
  ## Day-25
  ## Day-26
  ## Day-27
  ## Day-28
  ## Day-29
  ## Day-30
  
  

#### About  
This repository is to help people learn Deep Learning in 30 days. 
#### Contributors
[Pratyush Sethi](https://github.com/patty-13) , [Anshuman Singh](https://github.com/Anshuman-37)
