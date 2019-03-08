#include "ActivationFunctions.hpp"

float Relu::activation(const float &x) { return max(float(0.0), x); }
float Relu::derivativeactivation(const float &x) { return 0.0; }

float Sigmoid::activation(const float &x) { return 1.0 / (1.0 + exp(-x)); }
float Sigmoid::derivativeactivation(const float &x) { return x * (1.0 - x); }

float Tanh::activation(const float &x) {
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
float Tanh::derivativeactivation(const float &x) {
  return 1.0 - pow(activation(x), 2.0);
}

float Gauss::activation(const float &x) { return exp(-pow(x, 2.0)); }
float Gauss::derivativeactivation(const float &x) {
  return -2.0 * x * exp(-pow(x, 2.0));
}

float Bent::activation(const float &x) {
  return (sqrt(x * x + 1.0) - 1.0) / 2.0 + x;
}
float Bent::derivativeactivation(const float &x) {
  return x / (2.0 * sqrt(x * x + 1.0)) + 1.0;
}

float SoftPlus::activation(const float &x) { return log(1.0 + exp(x)); }
float SoftPlus::derivativeactivation(const float &x) {
  return 1.0 / (1.0 + exp(-x));
}

float Sinusoid::activation(const float &x) { return sin(x); }
float Sinusoid::derivativeactivation(const float &x) { return cos(x); }

float ISRLU::activation(const float &x) {
  if (x >= 0)
    return x;
  else
    return x / sqrt(1.0 + alpha * x * x);
}
float ISRLU::derivativeactivation(const float &x) {
  if (x >= 0)
    return 1.0;
  else
    return pow(1.0 / sqrt(1.0 + alpha * x * x), 3.0);
}
