#include "Layers.hpp"

Dropout::Dropout(float r, int oN, int iN, ofstream &logFile) {

  inputN = iN;
  outputN = oN;

  neuronDelta.resize(outputN);
  outputNeurons.resize(outputN);

  rate = r;

  type = LayerType::dropout;

  string msg = "Dropout layer created with rate " + to_string(rate);
  cout << msg << endl;
  if (logFile.is_open())
    logFile << msg << endl;
}

void Dropout::init() {

  for_each(weights.begin(), weights.end(),
           [&](auto &row) { fill(row.begin(), row.end(), 1.0); });

  actfunc = make_unique<Identity>();

  activation = [=](const float &input) { return actfunc->activation(input); };
  derivativeactivation = [=](const float &input) {
    return actfunc->derivativeactivation(input);
  };
}

void Dropout::fillInput(vector<float> &input) {

  cout << "Error: dropout as first layer. Aborting training." << endl;
  exit(-1);
}

void Dropout::calculateLayer(Layer &prevLayer) {

  if (outputNeurons.size() != prevLayer().size()) {
    cout << endl
         << "Error: dropout dimensions does not match with previous layer ("
         << outputNeurons.size() << " != " << prevLayer().size() << ")" << endl;
    cout << "Aborting training." << endl;
    exit(-1);
  }

  for (int iNeuron = 0; iNeuron < outputNeurons.size(); iNeuron++) {

    outputNeurons[iNeuron] = prevLayer()[iNeuron];
  }
}

void Dropout::rescaleWeights(const float &momentum, const float &rate,
                             Layer &prevLayer) {

  auto &pweights = prevLayer.getWeights();

  for_each(pweights.begin(), pweights.end(), [&](auto &row) {
    for_each(row.begin(), row.end(),
             [&](auto &val) { val = (dist(rng) + 0.5 < rate) ? 0.0 : val; });
  });
}

void Dropout::setNeuronDelta(const int &index, const int &target) {

  cout << "Error: dropout as last layer. Aborting training." << endl;
  exit(-1);
}

void Dropout::sumNeuronDelta(const int &index, const float &val) {

  neuronDelta[index] = val;
}

void Dropout::activateDelta(const int &index) {

  neuronDelta[index] *= derivativeactivation(outputNeurons[index]);
}

float Dropout::getWeightedSumNeuronDelta(const int &index) {
  return neuronDelta[index];
}