#include "main.hpp"

int main(int argc, char *argv[]) {

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <data Folder> <logFile> <saveFile>";
    return 0;
  }

  string dataFolder = argv[1];

  string trainImagesName = "/train-images.idx3-ubyte";
  string trainLabelsName = "/train-labels.idx1-ubyte";
  string testImagesName = "/t10k-images.idx3-ubyte";
  string testLabelsName = "/t10k-labels.idx1-ubyte";

  vector<vector<float>> trainImages;
  vector<vector<float>> testImages;

  // Reading MNIST test images into float vector from
  // dataFolder + testImagesName

  pMLP mlp = make_unique<MLP>(1);

  mlp->createLog(argv[2]);

  if (!mlp->readImages(dataFolder + testImagesName, testImages))
    return 0;

  // Reading MNIST train images into float vector from
  // dataFolder + trainImagesName

  if (!mlp->readImages(dataFolder + trainImagesName, trainImages))
    return 0;

  pPreprocessor preprocessor = make_unique<Preprocessor>();
  //   preprocessor->setVerbosity(1);

  preprocessor->normalize(trainImages, 28, 28);
  preprocessor->normalize(testImages, 28, 28);

  // Read MNIST label into float vector
  vector<float> trainLabels(trainImages.size());
  vector<float> testLabels(testImages.size());

  if (!mlp->readLabels(dataFolder + trainLabelsName, trainLabels))
    return 0;

  if (!mlp->readLabels(dataFolder + testLabelsName, testLabels))
    return 0;

  //   mlp->addLayer<Dense>(128, Activation::bent);
  //   mlp->addLayer<Dense>(64, Activation::isrlu);
  //   mlp->addLayer<Dense>(64, Activation::isrlu);
  mlp->addLayer<Dense>(392, Activation::isrlu);
  //   mlp->addLayer<Dense>(32, Activation::isrlu);
  mlp->addLayer<Dense>(10, Activation::tanh);

  mlp->compile(1e-4, 0.4);

  mlp->trainNetwork(trainImages, trainLabels, 5);

  mlp->validateNetwork(testImages, testLabels);

  mlp->saveNetwork(argv[3]);

  cout << "All done.\n";
}