//
//  NetworkTools.hpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/14/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#ifndef NetworkTools_hpp
#define NetworkTools_hpp

#include <stdio.h>


#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>



/* BaseNeuron:
 * This is a basic Neuron class which takes input as a vector and applys it's weights
 *
 * This can be as many dimensions as you want. best to specify in the constructor as nSize
 *
 */
template<typename classificationType>
class BaseNeuron{
public:
    BaseNeuron(int nSize){
        std::srand(0);
        // set the weights to the appropriate size
        weights.resize(nSize, 0.0);
        // randomize the weights between -1 and 1
        for( int i = 0; i < nSize; i++){
            float r = ( ( double ) std::rand() ) / RAND_MAX * 2 - 1;
            weights[i] = r;
        }
    }
    
    ~BaseNeuron(){}
    
    // Basic Neuron functionality
    virtual classificationType feedForward(std::vector<double> inputs){
        double s = 0.0;
        for ( int i = 0; i < size; i++){
            s += weights[i] * inputs[i];
        }
        return classify(s);
    }
    
    virtual classificationType classify( double sum ) = 0;
    
    std::vector<double> getWeights(){
        return weights;
    }
private:
    std::vector<double> weights;
    int size;
};

// Linear Neuron classifys based on +/- 1. Either above or below the line
class LinearNeuron : public BaseNeuron<int>{
    
    LinearNeuron(): BaseNeuron<int>(2){}
    
    int classify(double sum) override{
        if( sum > 0 )
            return 1;
        else
            return -1;
    }
};




#endif /* NetworkTools_hpp */
