//
//  BaseNetwork.hpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/14/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#ifndef BaseNetwork_hpp
#define BaseNetwork_hpp

#include <stdio.h>
#include "LinearAlgebraTools.hpp"

#include <vector>
#include <functional>
#include <cmath>

/*
 *
 *
 *
 */

class BaseNetwork{
    BaseNetwork(int nClasses):
        numClasses(nClasses)
    {
        
    }
    
    ~BaseNetwork(){
        
    }
    
    // Necessary function that maps to 0 1
    double activationFunc( double x ){
        return (std::tanh(x) + 1.0) / 2.0;
    }
    // Necessary Deriv of above function. Not this classes
    // job to make sure these are derivs of eachother
    virtual double activationFuncDeriv( double x){
        double tanhx = std::tanh(x);
        return (1.0 - tanhx*tanhx)/2.0;
    }
    // converts outputs to 1 index classes
    // ie.
    //          0 1 0     2
    //          1 0 0 ->  1
    //          1 0 0     1
    //          0 0 1     3
    Matrix<int> computeClasses(const Matrix<int>& output){
        size_t nRows = output.getRows();
        size_t nCols = output.getCols();
#ifdef DBG
        assert(nCols == numClasses);
#endif
        Matrix<int> classMatrix(nRows,1);
        for( size_t i = 0; i < nRows; i++ ){
            for( size_t j = 0; j < nCols; j++){
                if( output(i,j) != 0){
                    classMatrix(i, 0) = j+1;
                    break;
                }
            }
#ifdef DBG
            assert(classMatrix(i,0) != 0);
#endif
        }
        return classMatrix;
    }
    // inverse of function above
    Matrix<int> classesToOutput( const Matrix<int>& classes){
        size_t nRows = classes.getRows();
        Matrix<int> output(nRows, numClasses);
        output.fill(0);
        for( size_t i = 0; i < nRows; i++){
            output(i,classes(i, 0)+1) = 1;
        }
        return output;
    }
    
    // Applys the Activation Function to a matrix
    Matrix<double> applyActivationFunction( const Matrix<double>& x){
        // make a new matrix that is a copy of the input matrix
        Matrix<double> y(x.getRows(),x.getCols());
        y = x.applyFunction([](double x){return (std::tanh(x) + 1.0) / 2.0;});
        return y;
    }
    // apply deriv of function above
    Matrix<double> applyActivationFunctionDeriv( const Matrix<double>& x){
        Matrix<double> y(x);
        y = x.applyFunction([](double x){double tx = std::tanh(x); return (1.0 - tx*tx)/2.0;});
        return y;
    }
    
    
  //  Matrix<double> inputs;
    // dynamic weights
    Matrix<double> weights;
    // column of 1s
    Matrix<double> bias;
   // Matrix<double> outputs;
    Matrix<double> targetOuput;
    Matrix<double> targetClass;
    int numClasses;
};

#endif /* BaseNetwork_hpp */
