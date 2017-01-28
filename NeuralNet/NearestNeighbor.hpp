//
//  NearestNeighbor.hpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/15/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//


#ifndef NearestNeighbor_hpp
#define NearestNeighbor_hpp

#include <stdio.h>
#include <iostream>
#include "BaseClassifier.hpp"
#include "LinearAlgebraTools.hpp"

template<typename dataType>
class NearestNeighbor : public BaseClassifier<dataType>{
public:
  NearestNeighbor(){};
  ~NearestNeighbor(){};
  /* BaseClassifier function implementations*/
  /// For the kNN implimentation, train just saves the values
  void train(std::vector<dataType> input, std::vector<int> output) override{
    trainingInput = input;
    trainingOutput = output;
  };
  /// For the NearestNeighbor implimenation, predict compares each input to All
  /// outputs and selects the closest
  std::vector<int> predict(std::vector<dataType> intput) override{
    n_test = input.size();
    std::vector<int> yPredictions(n_test);
    for( int i = 0; i < n_test; i++){
      float distances =
    }
  };

  float getDistance( dataType in, dataType predicted ){
    return -1.0f;
  }

private:
  std::vector<dataType> trainingInput;
  std::vector<int>  trainingOutput;
}

// explicit specialization
template<>
float NearestNeighbor<Matrix<int>>::getDistance( Matrix<int> in, Matrix<int> pred){
  // L1
  in -= pred;
  in = in.applyFunction(std::abs);
  return in.sumMatrix();
  //L2
  /*
  in -= pred;
  in = in.applyFunction([](int x){return x*x;});
  return in.sumMatrix();
  */
}

#endif /*NearestNeighbor.hpp*/
