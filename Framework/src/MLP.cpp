
#include "MLP.hpp"

MLP::MLP(int verb) {

  verbosity = verb;
  dims.first = 0;
  dims.second = 0;

  pReader reader = make_unique<Reader>();

  reader->setVerbosity(verb);
}

void MLP::compile(const float &lR, const float &m) {

  learningRate = lR;
  momentum = m;
  Msg("Setting up the network...");
  for_each(layers.begin(), layers.end(), [&](pLayer &layer) { layer->init(); });

  Msg("Done.");
}

bool MLP::readImages(string fileName, vector<vector<float>> &vec) {

  pair<int, int> t_dims = reader->read_Mnist(fileName, vec);
  if (t_dims.first == -1 && t_dims.second == -1) {
    Msg("Error: open image file ", fileName);
    return false;
  } else if ((dims.first != 0 && dims.first != t_dims.first) ||
             (dims.second != 0 && dims.second != t_dims.second)) {
    Msg("Error: different image dimensions");
    return false;
  } else if (dims.first == 0 && dims.second == 0)
    dims = t_dims;

  if (verbosity > 0) {
    Msg("Number of images: ", vec.size());
    Msg("Dimension of images: ", vec[0].size(), " (", dims.first, "x",
        dims.second, ")");
  }
  return true;
}

bool MLP::readLabels(string filename, vector<float> &vec) {

  if (!reader->read_Mnist_Label(filename, vec)) {
    Msg("Error: reading train labels");
    return false;
  }

  if (verbosity > 0) {
    Msg("Number of labels: ", vec.size());
  }

  return true;
}

void MLP::createLog(string fileName) {
  logFile.open(fileName);
  if (logFile.is_open()) {
    Msg("Logfile", fileName, "created");
  }
}

void MLP::saveNetwork(string fileName) {

  saveFile.open(fileName);

  int ndropouts = 0;
  for_each(layers.begin(), layers.end(), [&](auto &l) {
    if (l->type == LayerType::dropout) {
      ndropouts++;
    }
  });

  saveFile << layers.size() - 1 << " " << momentum << " " << learningRate
           << endl;
  for_each(layers.begin(), layers.end(), [&](auto &l) {
    if (l->type == LayerType::dropout) {
      return;
    }
    saveFile << "#LB" << endl;
    saveFile << int(l->type) << " " << int(l->atype) << " " << l->getInputN()
             << " " << l->getOutputN() << endl;
    l->getWeights(saveFile);
    saveFile << "#LE" << endl;
  });

  saveFile << "#NE" << endl;

  saveFile.close();

  Msg("Saving trained network to", fileName, "complete.");
}

void MLP::loadNetwork(string fileName) {

  loadFile.open(fileName);
  if (!loadFile.is_open()) {
    Msg("Error: failed to load network from file", fileName);
    exit(-1);
  }

  int lnum;
  loadFile >> lnum >> momentum >> learningRate;
  for (int iLayer = 0; iLayer < lnum; iLayer++) {
    char id[3];
    loadFile >> id;
    if (id == "#NE")
      break;
    int ltype;
    int atype;
    int t_inputN;
    int t_outputN;
    loadFile >> ltype >> atype >> t_inputN >> t_outputN;
    auto e_atype = static_cast<Activation>(atype);
    if (ltype == 0)
      addLayer<Dense>(t_outputN, t_inputN, e_atype);

    vector<vector<float>> iWeights;
    for (int inp = 0; inp < t_inputN; inp++) {
      iWeights.push_back(vector<float>(t_outputN));
    }

    for (int inp = 0; inp < t_inputN; inp++) {
      for (int out = 0; out < t_outputN; out++) {
        loadFile >> iWeights[inp][out];
      }
    }

    layers.back()->loadWeights(iWeights);
    loadFile >> id;
  }

  loadFile.close();

  Msg("Loading trained network to", fileName, "complete.");
}

void MLP::trainNetwork(vector<vector<float>> &train_x, vector<float> &train_y,
                       int epochs, int batch_size) {

  auto start_train = std::chrono::high_resolution_clock::now();

  int nParams =
      accumulate(layers.begin(), layers.end(), 0, [&](int param, auto &layer) {
        return param += layer->getParamNum();
      });

  int nBatch = train_x.size() / batch_size;
  int nBatch_rem = train_x.size() % batch_size;

  int barWidth = 42;
  string eq;
  for (int i = 0; i < barWidth; ++i)
    eq += "=";

  Msg(eq);
  Msg("Start training network on", train_x.size(), "samples.");
  Msg("Number of trainable parameters:", nParams);
  Msg("Number of epochs:", epochs);
  Msg("Batch size:", batch_size, "(in", nBatch,
      "batch, the remainder:", nBatch_rem, ")");
  Msg(eq);

  if (batch_size == 1) {
    batch_size = train_x.size();
    nBatch = 1;
    nBatch_rem = 0;
  }
  cout << std::fixed;

  for (int iEpoch = 0; iEpoch < epochs; iEpoch++) {

    auto start_epoch = std::chrono::high_resolution_clock::now();

    float epochError = 0.0;
    float precision = 0.0;

    Msg("Epoch ", iEpoch + 1, "/", epochs, ":");

    int limit;

    for (int iBatch = 0; iBatch < nBatch; iBatch++) {

      if (iBatch + 1 == nBatch)
        limit = train_x.size();
      else
        limit = iBatch * batch_size + batch_size;

      for (int iImage = iBatch * batch_size; iImage < limit; iImage++) {

        float progress = 100 * float(iImage) / float(train_x.size());

        cout << iImage << " / " << train_x.size() << " [ ";

        int pos = barWidth * progress / 100;

        computeNetwork(train_x[iImage]);

        float target = train_y[iImage];

        float error = 0.0;

        for (int iNeuron = 0; iNeuron < (*layers.back())().size(); iNeuron++) {

          float t_error = ((int(target) == iNeuron) ? 1.0 : 0.0) -
                          layers.back()->getNeuronVal(iNeuron);

          error += pow(t_error, 2);

          layers.back()->setNeuronDelta(iNeuron, int(target));
        }

        for (int iLayer = layers.size() - 2; iLayer >= 0; iLayer--) {

          layers[iLayer]->resetNeuronDelta();

          for (int iPrevneuron = 0; iPrevneuron < layers[iLayer]->getOutputN();
               iPrevneuron++) {

            for (int iNeuron = 0; iNeuron < layers[iLayer + 1]->getOutputN();
                 iNeuron++) {
              layers[iLayer]->sumNeuronDelta(
                  iPrevneuron,
                  layers[iLayer + 1]->getWeightedSumNeuronDelta(iNeuron));
            }

            layers[iLayer]->activateDelta(iPrevneuron);
          }
        }

        for (int iLayer = layers.size() - 1; iLayer > 0; iLayer--) {

          layers[iLayer]->rescaleWeights(momentum, learningRate,
                                         *layers[iLayer - 1]);
        }

        epochError = error / (float(layers.back()->getOutputN()) + 1);

        auto stop_epoch = std::chrono::high_resolution_clock::now();
        auto elapsed_epoch = std::chrono::duration_cast<std::chrono::seconds>(
                                 stop_epoch - start_epoch)
                                 .count();
        auto elapsed_epoch_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop_epoch -
                                                                  start_epoch)
                .count();

        for (int i = 0; i < barWidth; ++i) {
          if (i < pos)
            cout << "=";
          else if (i == pos)
            cout << ">";
          else
            cout << "_";
        }

        cout << " ] " << int(progress) << " %  |  Error: ";

        auto eta = 100 * elapsed_epoch / progress - elapsed_epoch;

        if (elapsed_epoch_ms % 1000 == 0) {
          cout << std::setprecision(4);
          cout << epochError;
          cout << std::setprecision(1);
          cout << " , elapsed: " << elapsed_epoch << " s, eta: ";
          cout << eta << " s     ";
          cout << std::setprecision(4);
        }

        cout << "\r";
        cout.flush();
      }
    }

    auto stop_epoch = std::chrono::high_resolution_clock::now();
    auto elapsed_epoch = std::chrono::duration_cast<std::chrono::seconds>(
                             stop_epoch - start_epoch)
                             .count();
    Msg(train_x.size(), "/", train_x.size(), "[", eq,
        "] 100 % |  Error: ", epochError, ", elapsed:", elapsed_epoch,
        "s, eta: 0 s     ");

    // Msg("Mean square error: ", epochError);
  }

  auto stop_train = std::chrono::high_resolution_clock::now();
  auto elapsed_train =
      std::chrono::duration_cast<std::chrono::seconds>(stop_train - start_train)
          .count();
  Msg("Training complete in", elapsed_train, "s.");
}

bool MLP::computeNetwork(vector<float> &image, float label) {

  layers[0]->fillInput(image);

  for (int iLayer = 1; iLayer < layers.size(); iLayer++) {
    layers[iLayer]->calculateLayer(*layers[iLayer - 1]);
  }

  if (label == -1.0)
    return true;

  int predicted = layers.back()->getMostProbable();

  if (int(label) == predicted) {
    return true;
  } else
    return false;
}

void MLP::validateNetwork(vector<vector<float>> &test_x,
                          vector<float> &test_y) {
  int total = test_y.size();
  int passed = 0;

  int barWidth = 42;
  string eq;
  for (int i = 0; i < barWidth; ++i)
    eq += "=";

  Msg(eq);
  Msg("Start validating network on", test_x.size(), "samples.");
  Msg(eq);

  auto start_testing = std::chrono::high_resolution_clock::now();

  for (int iImage = 0; iImage < test_x.size(); iImage++) {

    cout << iImage << " / " << test_x.size() << " [ ";
    float progress = float(iImage) / float(test_x.size());
    int pos = barWidth * progress;

    if (computeNetwork(test_x[iImage], test_y[iImage])) {
      passed++;
    }

    for (int i = 0; i < barWidth; ++i) {
      if (i < pos)
        cout << "=";
      else if (i == pos)
        cout << ">";
      else
        cout << "_";
    }

    cout << " ] " << int(progress * 100.0) << " %\r";
    cout.flush();
  }

  auto stop_testing = std::chrono::high_resolution_clock::now();
  auto elapsed_testing = std::chrono::duration_cast<std::chrono::seconds>(
                             stop_testing - start_testing)
                             .count();

  Msg(test_x.size(), "/", test_x.size(), "[", eq, "] 100 %");

  Msg("Validation result: ", passed, "/", total, "passed (",
      float(passed) / float(total) * 100.0, "% accuracy),", elapsed_testing,
      "s elapsed");
}