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
#include <valarray>
#include <vector>
#include "BaseClassifier.hpp"
#include "LinearAlgebraTools.hpp"

template<typename dataType>
class NearestNeighbor : public BaseClassifier<dataType>{
public:
  NearestNeighbor(int k = 1){kNeighbor = k;};
  ~NearestNeighbor(){};
  int kNeighbor;
  /* BaseClassifier function implementations*/
  /// For the kNN implimentation, train just saves the values
  virtual void train(std::vector<dataType>& input, std::vector<int>& output) override{
    trainingInput = input;
    trainingOutput = output;
  };
  /// For the NearestNeighbor implimenation, predict compares each input to All
  /// outputs and selects the closest
  /// input k querys the k nearest neighbors and returns the most common
  virtual std::vector<int> predict(const std::vector<dataType>& input) override{
    int n_test = input.size();
    std::vector<int> yPredictions(n_test);

    for( int i = 0; i < n_test; i++){
      std::cout<<"PREDICTING "<<i<<std::endl;
      std::vector<float> distances(trainingInput.size());
      // Iter through ALL the training input... this is why this method is slow
      // and undesirable
      // also find the k closest input elements as we are going through
      std::vector<std::pair<int,float>> vIndDists;
      std::pair<int,float> curMaxIndDist;
      for( int j = 0; j < trainingInput.size(); j++){
        // get Distance (user defined metric) to this training data point
        float dist = getDistance(input[i],trainingInput[j]);
        distances[j] = dist;
        // if the vector of mins is empty, add this reguardless
        if( vIndDists.size() < kNeighbor ){
          vIndDists.emplace_back(j,dist);
          // invalidate Max
          curMaxIndDist.first = -1;
          curMaxIndDist.second = -1;
        }
        else{
          if( curMaxIndDist.second > dist){
            int indMax = curMaxIndDist.first;
            vIndDists[indMax].first = j;
            vIndDists[indMax].second = dist;
            // invalidate Max
            curMaxIndDist.first = -1;
            curMaxIndDist.second = -1;
          }
        }
        // recalc the max if we added any items
        if( curMaxIndDist.first == -1){
          for( auto pIndDist : vIndDists ){
            if( pIndDist.second > curMaxIndDist.second )
              curMaxIndDist = pIndDist;
          }
        }
      }
      if( kNeighbor == 1)
        yPredictions[i] = trainingOutput[vIndDists[0].first];
      else{
        std::vector<int> predictions, modePredictions;
        std::cout<<"A"<<std::endl;
        // now we have a vector of the k closest elements.
        // lets see what those elements had as labels
        predictions.clear();
        modePredictions.clear();
        std::cout<<"Cleared"<<std::endl;
        for ( auto pIndDist : vIndDists ){
          predictions.push_back(trainingOutput[pIndDist.first]);
        }

        std::sort(predictions.begin(), predictions.end());
        // now predictions is sorted. lets get the mode (or modes)

        int nModeCount = 0;
        int nCount = 1;
        int prevPred = -1;
        for( int iPred = 0; iPred < kNeighbor; iPred++){
          if( predictions[iPred] == prevPred )
            nCount++;
          else
            nCount = 1;
          if( nCount == nModeCount)
            modePredictions.push_back(predictions[iPred]);
          if( nCount > nModeCount){
            modePredictions.clear();
            modePredictions.push_back(predictions[iPred]);
            nModeCount = nCount;
          }
        }
        // now modePredictions contains the most common elements
        // most of the time this will be one item, but in the case
        // of multiple modes, multiple elements will be here
        // if there are many cases like this, may want to consider using
        // different distance metric
        // TODO: impliment a tie breaking function that reduces k until
        // the tie is resolved. This would require attaching distances to the
        // prediction vector. For now I will randomly pick one.
        if( modePredictions.size() == 1 )
          yPredictions[i] = modePredictions[0];
        else{
          std::cout<<"A TIE"<<std::endl;
          yPredictions[i] = modePredictions[std::rand() % modePredictions.size()];
        }
      }
    }
    std::cout<<"B"<<std::endl;
    return yPredictions;
  };

  // Note that distance, in any real impl. Should return positive
  float getDistance( dataType in, dataType predicted ){
    return -1.0f;
  }

private:
  std::vector<dataType> trainingInput;
  std::vector<int>  trainingOutput;
};

// explicit specialization for the matrix type. Note, if you choose to use a
// different type of NearestNeighbor, impliment your own getDistance function
template<>
float NearestNeighbor<Matrix<int>>::getDistance( Matrix<int> in, Matrix<int> pred){
  in -= pred;
  //L1
  in = in.applyFunction([](double x){return std::abs(x);});
  //L2
  //in = in.applyFunction([](int x){return x*x;});
  return in.sumMatrix();
}

template<>
float NearestNeighbor<std::vector<int>>::getDistance( std::vector<int> in, std::vector<int> pred ){
  assert(in.size() == pred.size());
  float sum = 0.0f;
  for( int i = 0; i < in.size(); i++){
    //L1
    in[i] = std::abs(in[i] - pred[i]);
    //L2
    //int val = in[i] - pred[i];
    //in[i] = val * val;
    sum += in[i];
  }
  return sum;
}

template<>
float NearestNeighbor<std::valarray<int>>::getDistance( std::valarray<int> in, std::valarray<int> pred ){
  in -= pred;
  // L1
  //in = std::abs(in);
  // L2
  in *= in;

  return in.sum();

}

#endif /*NearestNeighbor.hpp*/
