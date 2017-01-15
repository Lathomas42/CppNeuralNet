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

/* Matrix class
 *
 * This is a basic resizable matrix class that will come in handy in Neural Networks
 */
template< typename valType>
class Matrix{
public:
    Matrix(size_t nRows, size_t nCol):
        rows(nRows),
        columns(nCol),
        data(nRows*nCol)
    {
    }
    
    void setSize( size_t nRows, size_t nCol){
        rows = nRows;
        columns = nCol;
        data.resize(nRows*nCol);
    }
    
    size_t getRows(){
        return rows;
    }
    
    size_t getCols(){
        return columns;
    }
    
    void fill( valType val ){
        data.fill(val);
    }
    
    // getters and setters
    valType& operator()(size_t i, size_t j){
        return data[i*columns + j];
    }
    
    valType operator()(size_t i, size_t j) const{
        return data[i*columns + j];
    }

    valType getMultVal( Matrix<valType>& rhs, size_t i, size_t j){
        if( rhs.getCols() != getRows() )
            return valType(false);
        else{
            valType sum(0);
            for( size_t ind = 0; ind < rows; ind++){
                sum += operator()(i,ind) * rhs(ind,j);
            }
            return sum;
        }
    }
    
    Matrix<valType> operator*(Matrix<valType>& rhs){
        Matrix<valType> new_matrix(getCols(), rhs.getRows());
        for( size_t i = 0; i < getCols(); i++){
            for( size_t j = 0; j < rhs.getRows(); j++){
                rhs(i,j) = getMultVal(new_matrix,i,j);
            }
        }
        return new_matrix;
    }
    
    
private:
    size_t rows, columns;
    std::vector<valType> data;
};


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
