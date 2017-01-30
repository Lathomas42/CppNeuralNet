// Some helper functions to read CIFAR10 to valarrays or matrices

#include <vector>
#include <fstream>
#include <iostream>
#include <valarray>

struct CIFAR10Image{
  int classification;
  std::valarray<int> pixelValues;
};

class CIFAR10ImageSet{
public:
  CIFAR10ImageSet(){};
  //copy constructor
  CIFAR10ImageSet( const CIFAR10ImageSet& other){
    vClassifications = other.vClassifications;
    vPixelVals = other.vPixelVals;
  }
  CIFAR10ImageSet( std::vector<int> vClass, std::vector<std::valarray<int>> vPix):
    vClassifications(vClass),
    vPixelVals(vPix){ assert( vClass.size() == vPix.size());};

  CIFAR10ImageSet getSubset(int start = 0, int end = -1){
    if( end < 0 || end >= vClassifications.size())
      return CIFAR10ImageSet(*this);
    if( start < 0 )
      start = 0;
    std::vector<int> subClass( vClassifications.begin(), vClassifications.begin() + end);
    std::vector<std::valarray<int>> subPix( vPixelVals.begin(), vPixelVals.begin() + end);
    CIFAR10ImageSet subset( subClass, subPix);
    return subset;
  }
  size_t size(){
    assert( vClassifications.size() == vPixelVals.size());
    return vPixelVals.size();
  }
  std::vector<int> vClassifications;
  std::vector<std::valarray<int>> vPixelVals;
};

// download the CIFAR10 binary files from https://www.cs.toronto.edu/~kriz/cifar.html
// and put the relative path here
#define CIFAR10_DIRECTORY "../../cifar-10-batches-bin/"
// nFiles -1 will read all available files
CIFAR10ImageSet readCIFAR10File( std::string fname, int nFiles = -1 ){
  char buffer[3073];
  std::valarray<int> tmpVA(3072);
  std::string dir(CIFAR10_DIRECTORY);
  std::ifstream cifarBinaries (dir + fname, std::ios::in | std::ios::binary );
  CIFAR10ImageSet imgs;
  // while we can still read AND we are still reading files we want
  while( cifarBinaries.read(buffer,3073) && nFiles != 0){
    // buffer is full of 3073 bytes, lets turn that into a CIFAR10Image
    for( int i = 0; i < 3072; i++){
      tmpVA[i] = static_cast<uint8_t>(buffer[i+1]);
    }
    nFiles--;
    imgs.vClassifications.push_back(static_cast<uint8_t>(buffer[0]));
    imgs.vPixelVals.push_back(tmpVA);
  }
  cifarBinaries.close();
  return imgs;

}

class CIFAR10ImageSetIterator{
public:
  CIFAR10ImageSetIterator(std::string fname, int startingFile, int nFilesPerIter = 10):
    readPos(startingFile*3073),
    fn(CIFAR10_DIRECTORY + fname),
    nFPI(nFilesPerIter),
    cifarBinaries ( fn, std::ios::in | std::ios::binary){
          cifarBinaries.seekg(readPos);
  }
  ~CIFAR10ImageSetIterator(){
    cifarBinaries.close();
  }
  int readPos;
  const std::string fn;
  int nFPI;
  std::ifstream cifarBinaries;

  CIFAR10ImageSet nextSet(){
    char buffer[3073];
    std::valarray<int> tmpVA(3072);

    CIFAR10ImageSet imgs;
    int nFiles = nFPI;
    while( cifarBinaries.read(buffer,3073) && nFiles != 0){
      // buffer is full of 3073 bytes, lets turn that into a CIFAR10Image
      for( int i = 0; i < 3072; i++){
        tmpVA[i] = static_cast<uint8_t>(buffer[i+1]);
      }
      nFiles--;
      imgs.vClassifications.push_back(static_cast<uint8_t>(buffer[0]));
      imgs.vPixelVals.push_back(tmpVA);
      readPos += 3073;
    }
    return imgs;
  }
};
