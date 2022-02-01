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
  For Multi Layered Perceptron<br>
  <p>Neural Network uses optimizer to minimize the error in the algorithm and the error is actually computed is by using a         Loss Function which helps in identify how good or bad the model is performing.</p>
1. Loss Function <br> 
1.1 Regression Loss Function
  <ul>
    <em>
      <li>MEAN SQUARED ERROR</li>
      <li>MEAN SQAURED LOGRAITHMIC ERROR LOSS</li>
      <li>MEAN ABSOLUTE ERROR LOSS</li>
  </em>
  </ul>
1.2 Binary Classification Loss Function
  <ul>
    <em>
      <li>BINARY CROSS ENTROPY</li>
      <li>HINGE LOSS</li>
  </em>
  </ul>
1.3 Multi-Class Classification loss function
  <ul>
    <em>
      <li>CATEGORICAL CROSS ENTROPY LOSS</li>
      <li>KULLBACK LEIBLER DIVERGENCE LOSS</li>
  </em>
  </ul>
2. Optimization
  <ul>
    <em>
      <li>GRADIENT DESCENT</li>
      <li>STOCHASTIC GRADIENT DESCENT</li>
      <li>MINI BATCH GRADIENT DESCENT</li>
      <li>MOMENTUM</li>
      <li>NESTEROV ACCELERATED GRADIENT</li>
      <li>ADAGRAD</li>
      <li>ADADELTA</li>
      <li>ADAM</li>
      <li>RMSPROP</li>
  </em>
  </ul>
  
  
## Day 3 
   Code - For multi Layered Perceptron
   
   
## Day4
  Convolutional Neural Network  
   Theory - .... 
   



















#### About  
This repository is to help people learn Deep Learning in 30 days. 
#### Contributors
[Pratyush Sethi](https://github.com/patty-13) , [Anshuman Singh](https://github.com/Anshuman-37)
