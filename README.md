# Multi-layer perceptron in C++

My multilayer perceptron implementation in C++ to recognize handwritten numbers in the MNIST database.

## Environment

The code was developed and tested on Linux Mint 19.1, g++ version 7.3.0. Requirements:

* C++ compiler with C++14 support
* cmake

## Introduction

The [MNIST database](http://yann.lecun.com/exdb/mnist/) is loaded into `float` vectors and can be preprocessed with the `Preprocessor` class. In the current version only a simple normalization with the maximum RGB value is applied.

At the time being there are several types of the [activation functions](https://towardsdatascience.com/secret-sauce-behind-the-beauty-of-deep-learning-beginners-guide-to-activation-functions-a8e23a57d046):

* relu
* sigmoid
* tanh
* gauss
* bent
* softplus
* sinusoid
* inverse square root linear unit (ISRLU)

The number of layers is not limited. To add a new layer use one of the followings:

```C++
template <class ltype>
void MLP::addLayer(int outputN, int inputN, Activation afuncType);

template <class ltype> 
void MLP::addLayer(int outputN, Activation afuncType);
```

Example (the `inputN` argument is optional):

```C++
mlp->addLayer<Dense>(192, Activation::relu);
```

> Currently only the `Dense` type of layer is implemented.

The initialization of the network occurs with the `compile` function:

```C++
void MLP::compile(const float &learningRate, const float &momentum);
```

Finally the training and validation:

```C++
void MLP::trainNetwork(vector<vector<float>> &train_x, vector<float> &train_y,
                       int epochs, int nbatch = 1);

void MLP::validateNetwork(vector<vector<float>> &test_x, vector<float> &test_y);
```

In addition, there is a possibility to save/load the trained network:

```C++
void MLP::saveNetwork(string fileName);

void MLP::loadNetwork(string fileName);
```

---

### Run the examples

```bash
mkdir build && cd build && cmake ../
make -j4
./app/trainer ../data logfile_1 network_1
```

The trained network is saved into `network_1` which can be loaded and re-tested (or trained further):

```bash
./app/load ../data logfile_1_load network_1
```

## Example outputs

Several examples (training logs and the trained networks as well) can be found in the `example_outputs` folder. The best result so far  is achieved with the following layout (after 5 epochs, `momentum=0.4`, `learning rate=1e-4`):

1.  `mlp->addLayer<Dense>(192, 768, Activation::isrlu);`
1.  `mlp->addLayer<Dense>(10, Activation::tanh);`

Accuracy: 82.02%

## Used literature

* Multi-layer perceptron: [http://blog.refu.co/?p=931](http://blog.refu.co/?p=931)
* Commonly used activation functions: [https://en.wikipedia.org/wiki/Activation_function](https://en.wikipedia.org/wiki/Activation_function)
* Tensorflow/Keras: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)
* MNIST example 1: [https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d](https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d)
* MNIST example 2: [https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow#prerequisites](https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow#prerequisites)
* MNIST database: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* Read MNIST dataset in C++: [http://eric-yuan.me/cpp-read-mnist/](http://eric-yuan.me/cpp-read-mnist/)