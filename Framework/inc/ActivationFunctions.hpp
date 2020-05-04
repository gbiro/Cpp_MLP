#ifndef activationfunctions_h
#define activationfunctions_h

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

enum class Activation {
  relu,
  sigmoid,
  tanh,
  gauss,
  bent,
  softplus,
  sinusoid,
  isrlu,
  identity
};

class IActivationFunction {
public:
  virtual float activation(const float &x) = 0;
  virtual float derivativeactivation(const float &x) = 0;
};

class Identity : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Relu : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Sigmoid : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Tanh : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Gauss : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Bent : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class SoftPlus : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class Sinusoid : public IActivationFunction {
public:
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

class ISRLU : public IActivationFunction {
public:
  float alpha = 0.1;
  float activation(const float &x);
  float derivativeactivation(const float &x);
};

#endif