// Some helper functions to read CIFAR10 to valarrays or matrices

#include <vector>
#include <fstream>
#include <iostream>
#include <valarray>

struct CIFAR10Image{
  int classification;
  std::valarray<int> pixelValues;
};

// download the CIFAR10 binary files from https://www.cs.toronto.edu/~kriz/cifar.html
// and put the relative path here
#define CIFAR10_DIRECTORY "../../cifar-10-batches-bin/"
// nFiles -1 will read all available files
static std::vector<CIFAR10Image> readCIFAR10File( std::string fname, int nFiles = -1 ){
  char buffer[3073];
  std::string dir(CIFAR10_DIRECTORY);
  std::ifstream cifarBinaries (dir + fname, std::ios::in | std::ios::binary );
  std::vector<CIFAR10Image> imgs;
  // while we can still read AND we are still reading files we want
  while( cifarBinaries.read(buffer,3073) && nFiles != 0){
    // buffer is full of 3073 bytes, lets turn that into a CIFAR10Image
    CIFAR10Image img;
    img.classification = static_cast<uint8_t>(buffer[0]);
    img.pixelValues.resize(3072);
    for( int i = 0; i < 3072; i++){
      img.pixelValues[i] = static_cast<uint8_t>(buffer[i+1]);
    }
    nFiles--;
    imgs.push_back(img);
  }
  return imgs;

}
