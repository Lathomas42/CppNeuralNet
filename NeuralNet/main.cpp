//
//  main.cpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/14/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include <functional>
#include<thread>
#include<future>
#include <ctime>
#include "Tools/LinearAlgebraTools.hpp"
#include "Classifiers/NearestNeighbor.hpp"
#include "Tools/CIFAR10Reader.hpp"
#include "Classifiers/LinearClassifier.hpp"
double exampleFunc( double x){
    return std::sqrt(x);
}

#define TEST_KNN_MULTITHREAD
#define TEST_KNN
int main(int argc, const char * argv[]) {
    // insert code here...
#ifdef TEST_MATRICES
    Matrix<double> m2(3,3);
    m2.fill(2.0);
    std::cout<<"m2"<<std::endl;
    m2.operator<<(std::cout);

    Matrix<double> mI(3,3);
    mI.fill(0.0);
    mI(0,0) = 1.0;
    mI(1,1) = 1.0;
    mI(2,2) = 2.0;
    std::cout<<"mI"<<std::endl;
    mI.operator<<(std::cout);


    std::cout<<"Normal Mult"<<std::endl;
    Matrix<double> m3 = m2 * mI;
    m3.operator<<(std::cout);


    std::cout<<"Hadmard Mult"<<std::endl;
    m3 = m2.hadmardMult(mI);
    m3.operator<<(std::cout);

    std::cout<<"Kronker Mult"<<std::endl;
    m3 = m2.kroneckerMult(mI);
    m3.operator<<(std::cout);


    std::cout<<"HorzCat"<<std::endl;
    m3 = m2.horzcat(mI);
    m3.operator<<(std::cout);

    std::cout<<"vertCat"<<std::endl;
    m3 = m2.vertcat(mI);
    m3.operator<<(std::cout);

    std::cout<<"Add"<<std::endl;
    m3 = m2 + mI;
    m3.operator<<(std::cout);

    std::cout<<"Sub"<<std::endl;
    m3 = m2 - mI;
    m3.operator<<(std::cout);

    std::cout<<"take the square root"<<std::endl;
    m3 = m3.applyFunction(exampleFunc);
    m3.operator<<(std::cout);

    std::cout<<"Fill with random between -4 and 4"<<std::endl;
    m3.fillRandom(-4.0, 4.0);
    m3.operator<<(std::cout);

    std::cout<<"Sum of m3"<<std::endl;
    std::cout<<m3.sumMatrix()<<std::endl;
#endif
    std::clock_t    start;

#ifdef TEST_KNN
{
// this is an example of inline kNearestNeighbor use
    NearestNeighbor<std::valarray<int>> knn(1);

    {
      CIFAR10ImageSet trainImages = readCIFAR10File("test_batch.bin",1000);
      std::cout<<"Using Training set of: "<< trainImages.vClassifications.size()<<std::endl;

      knn.train(trainImages.vPixelVals, trainImages.vClassifications);
    }
    std::cout<<"Synchronous"<<std::endl;
    std::array<int,4> kValues = {1,3,5,10};
    for( auto k : kValues ){
      auto t_start = std::chrono::high_resolution_clock::now();
      CIFAR10ImageSetIterator imgIter("test_batch.bin",1000,500);
      knn.kNeighbor = k;
      bool done = false;
      int nC = 0;
      int nT = 0;
      CIFAR10ImageSet testBuffer;
      while (!done && nT < 8000){
        // get the next set of images
        imgIter.nextSet(testBuffer);
        if( testBuffer.size() != 0){
          // predict using the kNearestNeighbor
          std::vector<int> predicted(knn.predict(testBuffer.vPixelVals));
          for( int i = 0; i < testBuffer.size(); i++){
            nT++;
            if( predicted[i] == testBuffer.vClassifications[i])
              nC++;
          }
        }
        else{
          done = true;
        }
      }
      auto t_end = std::chrono::high_resolution_clock::now();

      std::cout<<" K VAL OF "<<k<<std::endl;
      std::cout << "Wall clock time passed: "
             << std::chrono::duration<double, std::milli>(t_end-t_start).count() <<std::endl;
      std::cout<<"% Correct at k Val of "<<k<<": "<<((float) nC)/nT<<std::endl;
    }
  }
#endif
#ifdef TEST_KNN_MULTITHREAD

{

// this is an example of inline kNearestNeighbor use
    NearestNeighbor<std::valarray<int>> knn(1);

    {
      CIFAR10ImageSet trainImages = readCIFAR10File("test_batch.bin",1000);
      std::cout<<"Using Training set of: "<< trainImages.vClassifications.size()<<std::endl;

      knn.train(trainImages.vPixelVals, trainImages.vClassifications);
    }
    std::cout<<"Asynchronous"<<std::endl;
    std::array<int,4> kValues = {1,3,5,10};
    for( auto k : kValues ){
      auto t_start = std::chrono::high_resolution_clock::now();

      CIFAR10ImageSetIterator imgIter("test_batch.bin",1000,500);
      knn.kNeighbor = k;
      bool done = false;
      int nC = 0;
      int nT = 0;

      std::vector<std::future<std::vector<int>>>promises;
      std::vector<std::vector<int>> vecOfClassifications;
      while (!done && nT < 8000){
        CIFAR10ImageSet testBuffer;
        // get the next set of images
        imgIter.nextSet(testBuffer);
        if( testBuffer.size() != 0){
          // predict using the kNearestNeighbor
          promises.push_back(std::async( &NearestNeighbor<std::valarray<int>>::predict, &knn, testBuffer.vPixelVals));
          nT += testBuffer.size();
          vecOfClassifications.push_back(testBuffer.vClassifications);
        }
        else
          done = true;
      }
      for( int i = 0; i < promises.size(); i++){
        std::vector<int> pred = promises[i].get();
        for( int j = 0; j < pred.size();j++){
          if( pred[j] == vecOfClassifications[i][j] )
            nC++;
        }
      }

      auto t_end = std::chrono::high_resolution_clock::now();

      std::cout<<" K VAL OF "<<k<<std::endl;
      std::cout << "Wall clock time passed: "
             << std::chrono::duration<double, std::milli>(t_end-t_start).count() <<std::endl;
      std::cout<<"% Correct at k Val of "<<k<<": "<<((float) nC)/nT<<std::endl;
    }
  }
#endif
    return 0;
}
