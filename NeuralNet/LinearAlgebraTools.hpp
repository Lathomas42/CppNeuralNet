//
//  LinearAlgebraTools.hpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/15/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#ifndef LinearAlgebraTools_hpp
#define LinearAlgebraTools_hpp

#include <stdio.h>
#include <vector>
/* Matrix class
 *
 * This is a basic resizable matrix class that will come in handy in Neural Networks
 */
template< typename valType>
class Matrix{
public:
    //default constructor
    Matrix():
        rows(0),
        columns(0),
        data(0)
    {}
    
    Matrix(size_t nRows, size_t nCol):
    rows(nRows),
    columns(nCol),
    data(nRows*nCol)
    {
    }
    
    // copy constructor
    Matrix( const Matrix<valType>& T):
        rows(T.getRows()),
        columns(T.getCols()),
        data(T.getRows()*T.getCols())
    {
        for( int i = 0; i < rows; i++){
            for( int j = 0; j < columns; j++){
                operator()(i,j) = T(i,j);
            }
        }
    }
    
    // copy operator
    Matrix<valType>& operator=(const Matrix<valType>& right){
        setSize( right.getRows(), right.getCols());
        for( int i = 0; i < rows; i++){
            for( int j = 0; j < columns; j++){
                operator()(i,j) = right(i,j);
            }
        }
        return *this;
    }
    
    /* --------------------------------------------------------
     * Basic Getters and Setters
     *
     */
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
    
    // element
    valType& operator()(size_t i, size_t j){
        return data[i*columns + j];
    }
    
    valType operator()(size_t i, size_t j) const{
        return data[i*columns + j];
    }
    
    valType getMultVal( const Matrix<valType>& rhs, size_t i, size_t j){
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
    
    /*-------------------------------------------
     * Assigment operators
     *
     */
    
    // replace with multiplication
    template<typename T>
    Matrix<valType>& operator*=(const T& mult){
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) *= mult;
            }
        }
        return *this;
    }

    // addition assigment
    template<typename T>
    Matrix<valType>& operator+=(const T& val){
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) += val;
            }
        }
        return *this;
    }
    
    // division assigment
    template<typename T>
    Matrix<valType>& operator/=(const T& val){
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) /= val;
            }
        }
        return *this;
    }
    
    // subtraction assigment
    template<typename T>
    Matrix<valType>& operator-=(const T& val){
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) -= val;
            }
        }
        return *this;
    }
    
    /* ------------------------------------------
     * Operations that produce a new matrix
     *
     */
    // constant multiplication
    template<typename T>
    Matrix<valType> operator*(const T& mult){
        Matrix<valType> copy(getRows(), getCols());
        copy *= mult;
        return copy;
    }
    
    // constant division
    template<typename T>
    Matrix<valType> operator/(const T& val){
        Matrix<valType> copy(getRows(), getCols());
        copy /= val;
        return copy;
    }
    
    // constant addition
    template<typename T>
    Matrix<valType> operator+(const T& val){
        Matrix<valType> copy(getRows(), getCols());
        copy += val;
        return copy;
    }
    
    // constant subtraction
    template<typename T>
    Matrix<valType> operator-(const T& val){
        Matrix<valType> copy(getRows(), getCols());
        copy -= val;
        return copy;
    }
    
    
    /* ------------------------------------------
     * Matrix Operations
     *
     */
    
    // matrix multiplication
    Matrix<valType> operator*(Matrix<valType>& rhs){
        Matrix<valType> new_matrix(getCols(), rhs.getRows());
        for( size_t i = 0; i < getCols(); i++){
            for( size_t j = 0; j < rhs.getRows(); j++){
                new_matrix(i,j) = getMultVal(rhs,i,j);
            }
        }
        return new_matrix;
    }
    
    // Hadamard multiplication
    /// Note: This requires matrices with the same coordinates
    Matrix<valType> hadmardMult( Matrix<valType>& rhs){
        Matrix<valType> new_matrix(getRows(), getCols());
        for( size_t i = 0; i < getCols(); i++){
            for( size_t j =  0; j < getRows(); j++){
                new_matrix(i,j) = operator()(i,j)*rhs(i,j);
            }
        }
        return new_matrix;
    }
    
    // Kroneker Multiplication
    Matrix<valType> kroneckerMult( Matrix<valType>& rhs ){
        Matrix<valType> kroneckerProduct( getRows()*rhs.getRows(), getCols()*rhs.getCols());
        for( size_t i = 0; i < getCols(); i++){
            for( size_t i2 = 0; i2< rhs.getCols(); i2++){
                for( size_t j = 0; j < getRows(); j++){
                    for( size_t j2 = 0; j2 < rhs.getCols(); j2++){
                    }
                }
            }
        }
    }
    
    
    // Transpose
    Matrix<valType> T(){
        Matrix<valType> new_matrix(getRows(), getCols());
        for( size_t i = 0; i < rows; i++){
            for( size_t j = 0; j < columns; j++){
                new_matrix(j,i) = operator()(i,j);
            }
        }
        return new_matrix;
    }
    
private:
    size_t rows, columns;
    std::vector<valType> data;
};


#endif /* LinearAlgebraTools_hpp */
