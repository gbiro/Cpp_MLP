#ifndef layers_h
#define layers_h

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "ActivationFunctions.hpp"

using namespace std;

typedef unique_ptr<IActivationFunction> pActFunc;

enum class LayerType { dense };

class Layer {
protected:
  int inputN, outputN;
  vector<vector<float>> weights;
  vector<vector<float>> prWeights;
  vector<float> outputNeurons;
  vector<float> neuronDelta;

  random_device rng;

  function<float(const float &)> activation;
  function<float(const float &)> derivativeactivation;
  pActFunc actfunc;

public:
  LayerType type;
  Activation atype;

  int getOutputN();
  int getInputN();
  void getWeights(ofstream &saveFile);

  void loadWeights(vector<vector<float>> iWeights);

  int getMostProbable();
  int getParamNum();
  float getNeuronVal(const int &index);
  void resetNeuronDelta();
  float activationFunction(const float &input);

  vector<float> &operator()() { return outputNeurons; }
  const vector<float> &operator()() const { return outputNeurons; }

  virtual void init() = 0;
  virtual void fillInput(vector<float> &input) = 0;
  virtual void calculateLayer(Layer &prevLayer) = 0;
  virtual void rescaleWeights(const float &momentum, const float &rate,
                              Layer &prevLayer) = 0;
  virtual void setNeuronDelta(const int &index, const int &target) = 0;
  virtual void sumNeuronDelta(const int &index, const float &val) = 0;
  virtual void activateDelta(const int &index) = 0;
  virtual float getWeightedSumNeuronDelta(const int &index) = 0;
};

class Dense : public Layer {

public:
  Dense(int outputN, int inputN, Activation afuncType, ofstream &logFile);
  uniform_real_distribution<float> dist{-0.5, 0.5};

  void init();
  void fillInput(vector<float> &input);
  void calculateLayer(Layer &prevLayer);
  void rescaleWeights(const float &momentum, const float &rate,
                      Layer &prevLayer);
  void setNeuronDelta(const int &index, const int &target);
  void sumNeuronDelta(const int &index, const float &val);
  void activateDelta(const int &index);
  float getWeightedSumNeuronDelta(const int &index);
};

#endif