# Deep Learning and Neural Networks

## Introduction

The subject of this report is deep learning and neural networks. This report is an attempt to give the reader - both technical and non-technical - a run through of what to expect in the presentation (we will talk about why we desperately need non-technical people in AI).

We first list down the basic definitions used in machine learning and deep learning. We then try to give a broad idea of what deep learning is and its variants i.e. the most popular and successful architectures in practice today.

We are trying to build the foundation for the presentation where the reader is able to follow along without any mental mapping or gaps. We have kept the code for the Python notebooks.

Deep learning admittedly for better or for worse has intricate mathematical foundations, which are not easy to understand without serious commitment. The reader is encouraged to dig into the references should she feel like it. For simplicity, code and notation is skipped in this report.

## Machine Learning Basics

Machine learning as defined by Mitchell - a computer program is said to learn from experience `E` with respect to some class of tasks `T` and performance measure `P`, if its performance at tasks in `T`, as measured by `P`, improves with experience `E`. It rhymes right.

The task here has a broad meaning and usually constitutes of problems which are not easily solved by hard computing. Following are a few tasks that machine learning algorithms tackle

* Classification:  In this type of task, the computer program is asked to specify which of `k` categories some input belongs to.

* Regression: In this type of task, the computer program is asked to predict a numerical value given some input.

* Transcription: In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe the information into discrete textual form.

* Machine translation: In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language.

* Structured output: Structured output tasks involve any task where the output is a vector with important relationships between the di?erent elements.

* Anomaly detection: In this type of task, the computer program sifts through a set of events or objects and ?ags some of them as being unusual or atypical.

* Synthesis and sampling: In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data.

The performance measure `P` is a quantitative measure of the correctness of the algorithm for task at hand `T`. It can be accuracy in case of classification, error rate etc. The performance measure is usually measured on something called a test set - that is separate from the data used for training the machine learning system.

The experience `E`: Machine learning algorithms can be broadly categorized as unsupervised or supervised by what kind of experience they are allowed to have during the learning process.

* Unsupervised learning algorithms experience a dataset containing many features, then learn useful properties of the structure of this dataset.

* Supervised learning algorithms experience a dataset containing features, but each example is also associated with a *label* or *target*.

Some machine learning algorithms do not just experience a ?xed dataset. For example, **reinforcement learning** algorithms interact with an environment, so there is a feedback loop between the learning system and its experiences.

### Capacity, Overfitting and Underfitting

The central challenge in machine learning is that our algorithm must perform well on new, previously unseen inputs - this is called generalization. We typically estimate the generalization error of a machine learning model by measuring its performance on a test set.  The error on the test set is called the test error and on the training set is called the train error. The factors determining how well a machine learning algorithm will perform are its ability to,

* Make the training error small - that is avoid *underfitting*
* Make the gap between training and test error small - that is avoid *overfitting*

A model?s capacity is its ability to fit a wide variety of functions.

Finally, we add a regularization term to our model, regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.

### Hyperparameters

Most machine learning algorithms have **hyperparameters**, settings that we can use to control the algorithm?s behavior. The values of hyperparameters are not adapted by the learning algorithm itself. One such example is the learning rate.

### Issues with Classical Machine Learning

The simple machine learning algorithms work well on a wide variety of important problems. They have not succeeded, however, in solving the central problems in AI, such as recognizing speech or recognizing objects.

* The Curse of Dimensionality: Many machine learning problems become exceedingly difficult when the number of dimensions in the data is high. This phenomenon is known as the curse of dimensionality.

## Neural Networks and Deep Learning

* Neural networks: A beautiful biologically-inspired programming paradigm which enables a computer to learn from observational data.
* Deep learning: A powerful set of techniques for machine learning using neural networks.

Neural networks and deep learning currently provide the best solutions to many problems in image recognition, speech recognition, and natural language processing. So let's learn about a simple neural network.

### Perceptron

Perceptron was developed in the 1950s and 1960s by the scientist Frank Rosenblatt, inspired by earlier work by Warren McCulloch and Walter Pitts.

A perceptron takes several binary inputs, x1,x2,..., and produces a single binary output. Neural networks are created by stacking single neurons. In a supervised learning setting, every neuron accepts the inputs from the input features - which then combine to form a complicated feature combination. Finally culminates into the output. Each layer is making more and more complex decisions to contribute in the output. The initial layers are making simple decisions and the later hidden layers are building on those decisions.

### The Architecture of NN

Usually, the leftmost layer in a network is called the input layer, which interacts with the input vector, and the neurons within the layer are called input neurons. The rightmost or output layer contains the output neurons, or, as in this case, a single output neuron. The middle layer is called a hidden layer.

Neural networks where the output from one layer is used as input to the next layer are called feedforward neural networks. This means there are no loops in the network - information is always fed forward, never fed back. There are other models of artificial neural networks in which feedback loops are possible. These models are called recurrent neural networks. We will talk about them later.

### Learning with Gradient Descent

What we'd like is an algorithm which lets us find weights and biases so that the output from the network approximates `y(x)` for all training inputs `x`. To quantify this, we have to come up with a loss function. Which could be for example the sum of squared errors of individual predictions (or MSE). In other words, we want to find a set of weights and biases which make the cost as small as possible. We'll do that using an algorithm known as gradient descent.

Let's suppose we're trying to minimize some function, `C(v)`. This could be any real-valued function of many variables, v=v1,v2,... and we'd like is to find where C achieves its global minimum. We'll also define the gradient of C to be the vector of partial derivatives. Using these derivatives, we will adjust our weights and biases. If we keep doing this, over and over, we'll keep decreasing C until - we hope - we reach a global minimum. The way the gradient descent algorithm works is to repeatedly compute the gradient, and then to move in the opposite direction, like "falling down" along the slope of the valley.

### Techniques for Improving Learning in a NN

* A better cost function called cross entropy loss function. The cross-entropy is positive, and tends toward zero as the neuron gets better at computing the desired output, `y`, for all training inputs, `x`. Which is what we want from a good loss function.

* L2 regularization: The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term.  The effect of regularization is to make it so the network prefers to learn small weights, all other things being equal. Large weights will only be allowed if they considerably improve the first part of the cost function. Regularization can be viewed as a way of compromising between finding small weights and minimizing the original cost function.

* L1 regularization: In this approach we modify the un-regularized cost function by adding the sum of the absolute values of the weights.

* Dropout: Dropout is a radically different technique for regularization. Unlike L1 and L2 regularization, dropout doesn't rely on modifying the cost function. Instead, in dropout we modify the network itself. We remove some of the neurons based on a probability.

* Data augmentation: Add more data by artificially creating it. Useful for computer vision applications.

## Modern Deep Learning Networks

### Deep Feedforward Networks

A type of NNs where the flow of information is from the input `x` through intermediate computations and finally to output `y`. There are no feedback connections in which outputs of the model are fed back into itself. They are a composition of many functions. The overall length of the composition is called the depth of the network. The ?nal layer of a feedforward network is called the output layer.

A feedforward network with a single layer is sufficient to represent any function, but the layer may be unfeasibly large and may fail to learn and generalize correctly. In many circumstances, using deeper models can reduce the number of units required to represent the desired function and can reduce the amount of generalization error.

### Convolutional Networks

Convolutional neural networks, or CNNs, are a specialized kind of neural network for processing data that has a known grid-like topology.

The name "convolutional neural network" indicates that the network employs a mathematical operation called convolution. Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

In convolutional network terminology, the first argument to the convolution is often referred to as the input, the second argument is called the kernel. The output is referred to as feature map.

A typical layer of a convolutional network consists of three stages. In the first stage, the layer performs several convolutions in parallel to produce a set of linear activations. In the second stage, each linear activation is run through a nonlinear activation function, such as the recti?ed linear activation function. This stage is sometimes called the detector stage. In the third stage, we use a pooling function to modify the output of the layer further.

The most successful application of CNNs can be seen in computer vision.

### Recurrent Nets

Recurrent neural networks, or RNNs are a family of neural networks for processing sequential data. These networks have a feedback loop feeding to the next layer the activations of the previous layer. Such a technique allows information sharing between layers. Information sharing allows the network to generate new sequences based on not only the current sequence but the sequences before the current sequence.

The architecture's output differs a lot depending on the application. For example,

* Many-to-many architecture where we output something at every time step (translation)
* Many-to-one architecture where we output something only at the end of the network (sentiment classification).
* One-to-many - in something like music generation.
* The kind of RNNs in a machine translation setting do not have same sequence lengths for input and output. The first part which takes the input sentence is called the encoder and the second part which takes the encoder and decodes it to the target language is called the decoder.

Some of the most popular RNNs are GRU or Gated Recurrent Unit, LSTM or Long Short Term Memory, Bi Directional RNN. The most successful application of RNNs are in NLP.

### Generative Adversarial Networks or GANs

It is a NN that is going to create sentences, create images, or generate anything. It is going to try and create thing which is very hard to tell the difference between generated stuff and real stuff.

One neural network, called the generator, generates new data instances, while the other, the discriminator, evaluates them for authenticity; i.e. the discriminator decides whether each instance of data it reviews belongs to the actual training dataset or not. This is essentially an actor-critic model. As the discriminator changes its behavior, so does the generator, and vice versa. Their losses push against each other.

The generator is going to try to keep getting better at fooling the discriminator into thinking that fake is real, and the discriminator is going to try to keep getting better at discriminating between the real and the fake.

## Deep Learning and Artificial General Intelligence

Deep learning is now the front runner technology in building components which can be part of artificial general intelligence. There are many components which we associate with AGI, some of which are discussed below and how deep learning is enabling us to make them efficiently. We end the discussion with what is the state of the art results for those components using deep learning.

### Learning to See

AI is expected to "see" the world in the sense we see. For this purpose, techniques using deep learning are enabling computer vision systems to see objects with human accuracy. The tasks involved are classifying the images for example whether something is a cat or a dog. Detecting objects in the image, creating bounding boxes around the objects in an image. The same techniques are applied to video as well. One of the most famous data sets for image recognition and computer vision is ImageNet. Most of the current models using deep learning are giving results in the close proximity of 99.9%.

One such example that we will see in the presentation is on ImageNet data set of 1.2 million images with 1000 classes which we represent as rank-3 tensors, or a 3D array. We use an NN architecture called resnet34 which is variant of a class of architectures called residual networks. The specifics are omitted in this report but it uses an incredibly powerful technique called transfer learning where a model is trained on images of a big data set like ImageNet and then used and then tuned on a smaller number of examples for the task at hand. We fetch a pre-trained model and then train it again on our own data set. Transfer learning has been hugely successful in computer vision tasks.

Another technique in computer vision is data augmentation, where we use the existing training examples and modify them while preserving their essence. The modified examples are then added back to the training set and model gets is trained on the 'augmented' data. One such example is taking the images in a training set, rotate them horizontally and add them back to the training set.

CNNs and its variants are now the workhorse of computer vision. They are identifying from trivial Cats vs Dogs images to cancers. From cars on the street to humans crossing the road. A lot of experiments in identifying terrain in satellite images have also been successful.

### Learning to Analyze

AI is expected to carry out or automate or simplify a lot of the manual analysis done by humans. One of the most common data formats available is the tabular data found in data bases, Excel files, CSV files etc.

As we shall see in the presentation, deep learning has been incredibly successful in this area. The results can be gauged from the winners of Kaggle competitions who have used deep learning on tabular data instead of using classical machine learning techniques. One reason of this success is the simplicity of implementation provided by the modern libraries and almost non-existent feature engineering. Earlier machine learning models relied heavily on feature engineering which was identifying combinations or inventing new features manually or by having intuition into the domain. Also, deep learning does not make any assumptions about the shape of the hypothesis which comes in handy when trying to fit the data.

The rule of thumb is if there is a pattern and there is sufficient data, the network will learn those patterns. Needless to say that the mind of the network is susceptible to the bias in the data and might over fit. The usual process of implementing a model to predict from tabular data goes through a lot of trial and error.

Most of the time spent in getting a NN model to work for tabular data goes into data engineering. That is, cleaning the data for the model to understand. Adding suitable date related fields, sometimes manually identifying what features might be useful and helping the network as a result. Special care is given to string values such as nouns, which do not have a meaningful numerical value such as custodian accounts. These are called categorical values and are represented using a technique called one hot encoding. It usually ends up bloating the dimensionality of the input data but this is the best we have right now.

### Learning to Understand

AI is expected to interact with the world around verbally. In order to do that, it needs to understand natural language. Not just one language, any human language. In this task, deep learning started getting attention only from 2012. The techniques and tools to solve NLP problems using deep learning is at least two-to-three years behind computer vision. Nevertheless, deep learning has still surpassed all the other techniques of solving NLP problems in 2016.

Some of the tasks in NLP where deep learning is used is language modelling, machine translation, sentiment analysis and collaborative filtering (recommendation systems). 

Language modelling is building a model which can understand and speak a human language. For example, given a text script of Star Wars, can a model learn the style and start speaking like the script of Star Wars. As we shall see in the presentation, this is indeed possible. The model is expected to learn the nuances of the language, i.e. the vocabulary and the grammar.

Machine translation is building a model which can translate from one human language to another given sufficient example translations. Usually around a million examples. Having such a model and data can help building something like C3PO which was trained in 6 million languages.

Sentiment analysis is classification problem dealing with identifying whether a text document has a positive or negative sentiment. It uses models for language modelling as a sub component.

Collaborative filtering is the fancily named technique for building recommendation systems. A human can easily identify whether two objects are similar or not. AI is also expected to do that. As we shall see in the presentation using deep learning we can make state of the art recommendation systems.

There are many other applications of deep learning to list exhaustively here. But the overarching technique that powers them is the success of sequence to sequence models. And the NN architecture behind sequence to sequence model is recurrent neural networks or RNNs and its variants which we saw earlier in this report.

One of the challenges in RNNs is they are hard to train and take a long time. Also, the concept of transfer learning is slowly making its way into NLP, which will make training models as fast as in computer vision. Hopefully, we will have faster ways to train RNNs. It's just a matter of time.

As of now, for almost all the NLP tasks, deep learning is the best bet for state of the art results.

### Learning to Create

To create is to be immortal. Humanity survives through its creation. AI is also expected to create. Deep learning is finally enabling AI to create things. Things like images, videos, music etc. There is no technical limitation as to what it can create for example, a zebra of the height of a horse. Zebras as big as horses do not exist but AI can create such images with incredible detail.

As we shall see in the presentation, with a NN architecture called generative adversarial networks or GANs, we can give the model some specific data set where it can learn and start generating pictures which might belong to the entities in that set. For example, celebrities faces or some specific scenery.

There is variation of GANs called cycle GANs which is a combination of two models, a generator and a discriminator training each other and getting better and better at its task. Using cycle GANs, we can create DSLR like photos of Monet paintings knowing the fact that we don't have the photos of landscapes Monet painted. Or generate Carrie Fisher in a Star Wars movie knowing she is not there anymore to play the part :(.

The deep learning community is calling GANs as "the coolest idea in deep learning in the last 20 years". The possibilities are endless. And scary.

## The State of AI

For a long time we have been programming computers to do our tasks. The tasks we know, but for the tasks we ourselves don't know how to do correctly, how do we program a computer to do it? Deep learning allows the computers to learn on its own how to do the tasks we don't know how to do ourselves. Many companies which affect our lives on a daily basis use machine learning - Google search for example.

Using deep learning, computers can learn. Better than us on some things which has lead to breakthroughs like automatic drug discovery without domain knowledge using relevant data and computation.

Using deep learning, computers can listen. Almost as good as native speakers. Computers are now better than people on traffic recognizing signs. Not only that, Google mapped every street number in 2 hours which would have taken years.

Computers can now understand complex sentences- negative and positive with near human performance. Computers can read and write describing pictures close to human performance

So how will it affect us? I don't know.

It will take 300 years in the developing world to train enough people for complex task such as radiology. So we need AI there to augment human knowledge. But there are a lot of people who are doing tasks like reading and writing - speaking and listening - looking at things and identifying. Anything a person can do with less than a second of thinking - we can use AI to automate it (not a perfect rule). Built by people who are no more qualified than you and me.

ML revolution, unlike the industrial revolution will never settle down - what's possible is difficult to estimate. So the question is,

How will we structure our society? When AI is the new electricity.

## Incorporating AI in Existing Organizations

In this era of data, truly defensible technologies can be successfully created if you have data. A million strong image set of a particular kind of cancer that your company can identify will make it really effective. And hard to replicate. The algorithms can be easily replicated but not the data. This is what it means to be an AI company. A traditional company and few neural networks is not an AI company.

Truly successful companies will tend to have strategic data acquisition plans. Even if transfer learning can really help in training good models without much data, the companies which has the most relevant data will win. The MVP can be launched with smaller data set which can be used to acquire more users which generates more data. This has been one of the more successful ways of bootstrapping AI companies.

Data warehousing accessible by all the BUs for creative usage of data is another way successful AI companies envision and launch products. Developers working in silos needing 50 different approvals before getting to use the data has not been a successful strategy.

Looking for automation opportunities knowing what is possible is driving the application of AI. In this regard, what if you want to integrate AI for such automation in your company, few pointers are listed below

* Make a centralized AI team even if you sell gift cards.

* Hire AI talent for this team and let them work creatively with different BUs to potentially solve their problems.

* Make it visible company-wide that you are looking into AI for solutions.

* Train the existing workforce - in diverse BUs. Opportunities will come up.

## The Ethics of AI and Biases in People Building AI Systems

We don't have a comprehensive set of rules here, but consider these examples

* Images of people of color were classified as "Gorilla".
* When "She is a doctor" and "He is a nurse" was translated to Turkish (it has no gender pronouns) from English and translated back, it became "She is a nurse" and "He is a doctor".

Both these examples are from Google.

Predictive policing systems from companies like Palantir sends more police to one neighborhood, which leads to more arrests. Which feeds back into the system as a more crime prone area. The system as a result sends more police. A vicious cycle.

The error rate of face detection systems by IBM and Face++ is 46.8% when it comes to people of color. Which is needless to say abysmal. A product management failure where the product simply doesn't work for a large percentage of people.

Facebook fired human editors a while back and the algorithm immediately posts fake news in the Trending feature. Which was then quoted by a real world leader thanking Facebook for the "news". And consequently making decisions on it.

More and more examples can be found about people living in their "bubble". They like something, the algorithm picks it up, suggests the same things and the loop continues.

Meetup made a decision to promote more technical content to women because their collaborative filtering algorithm was suggesting technical meetups only to men. Meetups attended by men, the feedback loop picking it up and then suggesting more and more such meetups to men. The company recognized this and fixed it. Not all companies do that.

The AI systems we build affects real life. Real people. We hardly ever asked the question whether the systems we are making has a societal impact. Now we have to.

So what to do? I am of the opinion that people do dumb stuff because they don't know any better. If given the right information and tools, they do make the right decisions. Some of the ways are

* Provide easy and accessible tools to many people - through online education. People of all kinds of background working on their own domain having access to AI.
* Hiring people of diverse background.
* Because diverse teams perform better. Period.
* Good interview practices to promote diversity.

Not doing this can sometimes lead to incredibly difficult situations for you and your company. In the end, the code you write is your responsibility. Sometimes even making you look bad in history, for example IBM was the company behind the technical infrastructure ("Deaths Calculator") of the holocaust.

The universities are taking a big step in this regard. Most of them are adding ethics courses for students to take. The questions are real. The problems are real. We all need to do better.

## The Limitations of AI and Open Questions

* AI systems reflect the biases in the data. There is a dire need to be careful and test.

* How can the reliability of a critical AI system (a self-drive car) be verified?

* When a system makes a bad decision - who is responsible? The designer, the data collector, the product vendor?

* Should decisions by AI systems be explainable? Why my application was rejected for example.

* AI in information filtering and ranking - it reflects our taste and creates a bubble. Too much filtering can lead to censorship and too little may let through illegal content.

Below are the links to the notebooks to co-relate the information in this report.

## Links to Notebooks

## References

1. [Deep Learning Coursera](https://www.deeplearning.ai/)
1. [Deep Learning Book](http://www.deeplearningbook.org)
1. [Fast.AI](http://www.fast.ai/)
1. [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)

---
