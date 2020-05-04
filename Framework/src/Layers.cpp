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

vector<vector<float>> &Layer::getWeights() { return weights; }

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
