#include "main.hpp"

int main(int argc, char *argv[]) {

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <data Folder> <logFile> <loadFile>";
    return 0;
  }

  string dataFolder = argv[1];

  string testImagesName = "/t10k-images.idx3-ubyte";
  string testLabelsName = "/t10k-labels.idx1-ubyte";

  vector<vector<float>> testImages;

  pMLP mlp = make_unique<MLP>(1);

  mlp->createLog(argv[2]);

  mlp->loadNetwork(argv[3]);

  if (!mlp->readImages(dataFolder + testImagesName, testImages))
    return 0;

  pPreprocessor preprocessor = make_unique<Preprocessor>();
  //   preprocessor->setVerbosity(1);

  preprocessor->normalize(testImages, 28, 28);

  //   read MNIST label into float vector
  vector<float> testLabels(testImages.size());

  if (!mlp->readLabels(dataFolder + testLabelsName, testLabels))
    return 0;

  mlp->validateNetwork(testImages, testLabels);

  cout << "All done.\n";
}