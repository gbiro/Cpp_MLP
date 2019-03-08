
#include "Preprocessor.hpp"

void Preprocessor::setVerbosity(int v) { verbosity = v; }

void Preprocessor::normalize(vector<vector<float>> &images, int dimx,
                             int dimy) {

  for_each(images.begin(), images.end(), [&](vector<float> &image) {
    float max = *max_element(image.begin(), image.end());
    float min = *min_element(image.begin(), image.end());
    float normfact = max - min;
    for_each(image.begin(), image.end(),
             [&](float &pixel) { pixel /= normfact; });
  });

  if (verbosity > 0) {
    cout << "The first (normalized) image: " << endl;
    for (int x = 0; x < dimx; x++) {
      for (int y = 0; y < dimy; y++) {
        int coord = y + x * dimy;
        if (images[0][coord] == 0)
          cout << " ";
        else
          cout << "x";
      }
      cout << "\n";
    }
  }
}

void Preprocessor::average(vector<vector<float>> &images, int dimx, int dimy) {

  int picNum = images.size();
  vector<float> avgPic(dimx * dimy);
  fill(avgPic.begin(), avgPic.end(), 0.0);

  for (int iPic = 0; iPic < picNum; iPic++) {
    for (int x = 0; x < dimx; x++) {
      for (int y = 0; y < dimy; y++) {
        int coord = y + x * dimy;
        avgPic[coord] += images[iPic][coord] / picNum;
      }
    }
  }

  for (int iPic = 0; iPic < picNum; iPic++) {
    for (int x = 0; x < dimx; x++) {
      for (int y = 0; y < dimy; y++) {
        int coord = y + x * dimy;
        if (avgPic[coord] == 0.0)
          continue;
        images[iPic][coord] /= avgPic[coord];
      }
    }
  }

  if (verbosity > 0) {
    cout << "The first (averaged) image: " << endl;
    for (int x = 0; x < dimx; x++) {
      for (int y = 0; y < dimy; y++) {
        int coord = y + x * dimy;
        if (images[0][coord] == 0)
          cout << " ";
        else
          cout << "X";
      }
      cout << "\n";
    }
  }
}