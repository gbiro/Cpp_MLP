#ifndef mlp_h
#define mlp_h

#include "Layers.hpp"
#include "Reader.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace std;

typedef unique_ptr<Reader> pReader;
typedef unique_ptr<Layer> pLayer;

class MLP {
private:
  int verbosity;
  pair<int, int> dims;

  vector<pLayer> layers;

  pReader reader;

  ofstream logFile;
  ofstream saveFile;
  ifstream loadFile;

  float learningRate, momentum, lmse;

public:
  MLP(int verbosity = 0);

  template <class ltype>
  void addLayer(int outputN, int inputN, Activation afuncType);

  template <class ltype> void addLayer(int outputN, Activation afuncType);

  bool readImages(string filename, vector<vector<float>> &vec);
  bool readLabels(string filename, vector<float> &vec);

  void compile(const float &learningRate, const float &momentum);

  void trainNetwork(vector<vector<float>> &train_x, vector<float> &train_y,
                    int epochs, int nbatch = 1);

  bool computeNetwork(vector<float> &image, float label = -1.0);

  void validateNetwork(vector<vector<float>> &test_x, vector<float> &test_y);

  void createLog(string fileName);
  void saveNetwork(string fileName);
  void loadNetwork(string fileName);

  template <typename Arg> void Msg(const Arg &arg) {
    cout << arg << endl;
    if (logFile.is_open()) {
      logFile << arg << endl;
    }
  }

  template <typename Arg, typename... Args>
  void Msg(const Arg &arg, const Args &... args) {
    cout << arg << " ";
    if (logFile.is_open()) {
      logFile << arg << " ";
    }
    Msg(args...);
  }
};

template <class ltype>
void MLP::addLayer(int outputN, int inputN, Activation afuncType) {
  if (layers.size() == 0 || layers.back()->getOutputN() == inputN)
    layers.push_back(make_unique<ltype>(outputN, inputN, afuncType, logFile));
  else {
    Msg("Error: creating", layers.size() + 1, ". layer: input dimensions");
  }
}

template <class ltype> void MLP::addLayer(int outputN, Activation afuncType) {

  if (layers.size() != 0)
    layers.push_back(make_unique<ltype>(outputN, layers.back()->getOutputN(),
                                        afuncType, logFile));
  else if (dims.first != 0 && dims.second != 0)
    layers.push_back(make_unique<ltype>(outputN, dims.first * dims.second,
                                        afuncType, logFile));
  else {
    Msg("Error: creating", layers.size() + 1, ". layer: input dimensions");
  }
}

#endif