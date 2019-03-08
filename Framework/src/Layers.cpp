#include "Layers.hpp"

int Layer::getOutputN() { return outputN; }

int Layer::getInputN() { return inputN; }

void Layer::getWeights(ofstream &saveFile) {

  for (int inp = 0; inp < inputN; inp++) {
    for (int out = 0; out < outputN; out++) {
      saveFile << weights[inp][out] << " ";
    }
    saveFile << endl;
  }
}

void Layer::loadWeights(vector<vector<float>> iWeights) { weights = iWeights; }

int Layer::getParamNum() {

  int params = 0;
  for (auto &row : weights) {
    params += row.size();
  }
  return params;
}

float Layer::getNeuronVal(const int &index) { return outputNeurons[index]; }

int Layer::getMostProbable() {

  return distance(outputNeurons.begin(),
                  max_element(outputNeurons.begin(), outputNeurons.end()));
}

float Layer::activationFunction(const float &input) {

  return activation(input);
}

void Layer::resetNeuronDelta() {

  fill(neuronDelta.begin(), neuronDelta.end(), 0);
}

Dense::Dense(int oN, int iN, Activation afunc, ofstream &logFile) {

  inputN = iN;
  outputN = oN;

  type = LayerType::dense;
  atype = afunc;

  //   weights.resize(inputN * outputN);
  for (int inp = 0; inp < inputN; inp++)
    weights.push_back(vector<float>(outputN));

  neuronDelta.resize(outputN);
  outputNeurons.resize(outputN);

  string s_afunc;

  if (afunc == Activation::relu) {
    actfunc = make_unique<Relu>();
    s_afunc = "Relu";
  } else if (afunc == Activation::sigmoid) {
    actfunc = make_unique<Sigmoid>();
    s_afunc = "Sigmoid";
  } else if (afunc == Activation::tanh) {
    actfunc = make_unique<Tanh>();
    s_afunc = "Tanh";
  } else if (afunc == Activation::gauss) {
    actfunc = make_unique<Gauss>();
    s_afunc = "Gauss";
  } else if (afunc == Activation::bent) {
    actfunc = make_unique<Bent>();
    s_afunc = "Bent";
  } else if (afunc == Activation::softplus) {
    actfunc = make_unique<SoftPlus>();
    s_afunc = "SoftPlus";
  } else if (afunc == Activation::sinusoid) {
    actfunc = make_unique<Sinusoid>();
    s_afunc = "Sinusoid";
  } else if (afunc == Activation::isrlu) {
    actfunc = make_unique<ISRLU>();
    s_afunc = "ISRLU";
  } else {
    exit(-1);
  }

  activation = [=](const float &input) { return actfunc->activation(input); };
  derivativeactivation = [=](const float &input) {
    return actfunc->derivativeactivation(input);
  };

  string msg = "Dense layer created with I/O dimensions " + to_string(inputN) +
               " " + to_string(outputN) +
               ", the activation function: " + s_afunc;
  cout << msg << endl;
  if (logFile.is_open())
    logFile << msg << endl;
}

void Dense::init() {

  for_each(weights.begin(), weights.end(), [&](auto &row) {
    generate(row.begin(), row.end(), [&]() { return dist(rng); });
  });

  prWeights = weights;
}

void Dense::fillInput(vector<float> &input) {

  for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {
    float val = 0.0;
    for (int iInput = 0; iInput < input.size(); iInput++) {
      val += input[iInput] * weights[iInput][iNeuron];
    }
    outputNeurons[iNeuron] = activationFunction(val);
  }
}

void Dense::calculateLayer(Layer &prevLayer) {

  for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {
    float val = 0.0;

    for (int iPrev = 0; iPrev < prevLayer().size(); iPrev++) {
      val += prevLayer()[iPrev] * weights[iPrev][iNeuron];
    }

    outputNeurons[iNeuron] = activationFunction(val);
  }
}

void Dense::rescaleWeights(const float &momentum, const float &rate,
                           Layer &prevLayer) {

  vector<vector<float>> t_weights = weights;
  for (int inp = 0; inp < inputN; inp++) {

    for (int out = 0; out < outputN; out++) {
      weights[inp][out] = weights[inp][out] +
                          momentum * (weights[inp][out] - prWeights[inp][out]) +
                          rate * neuronDelta[out] * prevLayer.getNeuronVal(inp);
    }
  }
  prWeights = t_weights;
}

void Dense::setNeuronDelta(const int &index, const int &target) {

  neuronDelta[index] =
      (((target == index) ? 1.0 : 0.0) - outputNeurons[index]) *
      derivativeactivation(outputNeurons[index]);
}

void Dense::sumNeuronDelta(const int &index, const float &val) {

  neuronDelta[index] += val;
}

void Dense::activateDelta(const int &index) {

  neuronDelta[index] *= derivativeactivation(outputNeurons[index]);
}

float Dense::getWeightedSumNeuronDelta(const int &index) {

  float sum = 0;
  for (int inp = 0; inp < inputN; inp++) {
    sum += neuronDelta[index] * weights[inp][index];
  }
  return sum;
}