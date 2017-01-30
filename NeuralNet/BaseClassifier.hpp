//
//  BaseClassifier.hpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/15/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//


#ifndef BaseClassifier_hpp
#define BaseClassifier_hpp

#include <vector>

template <typename inValType>
class BaseClassifier{
  /// This funciton takes in
  /// input: N x D where D is the dims of valType.
  /// note valType can me Matrix<double> or anything else
  virtual void train(std::vector<inValType>& input, std::vector<int>& output) = 0;
  /// This function takes in an input dataset and outputs the classifications
  virtual std::vector<int> predict(const std::vector<inValType>& intput) = 0;

};

#endif /*BaseClassifier.hpp*/
