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
#include <iostream>
#include <algorithm>
// this is a debug flag
#define DBG

#ifdef DBG
#include <assert.h>
#endif

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
        std::fill(data.begin(),data.end(),val);
    }
    
    // element
    valType& operator()(size_t i, size_t j){
#ifdef DBG
        assert(i < rows && j < columns);
#endif
        return data[i*columns + j];
    }
    
    valType operator()(size_t i, size_t j) const{
#ifdef DBG
        assert(i < rows && j < columns);
#endif
        return data[i*columns + j];
    }
    
    valType getMultVal( const Matrix<valType>& rhs, size_t i, size_t j){
#ifdef DBG
        assert(getCols() == rhs.getRows() && i < getCols() && j < rhs.getRows());
#endif
        valType sum(0);
        for( size_t ind = 0; ind < rows; ind++){
            sum += operator()(i,ind) * rhs(ind,j);
        }
        return sum;
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
        Matrix<valType> copy(*this);
        copy *= mult;
        return copy;
    }
    
    // constant division
    template<typename T>
    Matrix<valType> operator/(const T& val){
        Matrix<valType> copy(*this);
        copy /= val;
        return copy;
    }
    
    // constant addition
    template<typename T>
    Matrix<valType> operator+(const T& val){
        Matrix<valType> copy(*this);
        copy += val;
        return copy;
    }
    
    // constant subtraction
    template<typename T>
    Matrix<valType> operator-(const T& val){
        Matrix<valType> copy(*this);
        copy -= val;
        return copy;
    }
    
    
    /* ------------------------------------------
     * Matrix Assignment Operators
     *
     */
    
    // basic matrix addition assigment operator
    Matrix<valType>& operator+=(const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) += rhs(i,j);
            }
        }
        return *this;
    }
    
    // basic matrix subtraction assigment operator
    Matrix<valType>& operator-=(const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
        for( size_t i=0; i < getRows(); i++){
            for(size_t j = 0; j < getCols(); j++){
                operator()(i,j) -= rhs(i,j);
            }
        }
        return *this;
    }

    /* ------------------------------------------
     * Matrix Operators that produce a new matrix of the same size
     *
     */
    
    // basic matrix addition
    Matrix<valType> operator+(const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
        Matrix<valType> copy(*this);
        copy += rhs;
        return copy;
    }
    
    // basic matrix subtraction
    Matrix<valType> operator-(const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
        Matrix<valType> copy(*this);
        copy -= rhs;
        return copy;
    }
    
    
    /* Matrix Multiplication types
     *
     *
     */
    // standard matrix multiplication
    Matrix<valType> operator*( const Matrix<valType>& rhs){
#ifdef DBG
        assert(getCols() == rhs.getRows());
#endif
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
    Matrix<valType> hadmardMult( const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
        Matrix<valType> new_matrix(getRows(), getCols());
        for( size_t i = 0; i < getCols(); i++){
            for( size_t j =  0; j < getRows(); j++){
                new_matrix(i,j) = operator()(i,j)*rhs(i,j);
            }
        }
        return new_matrix;
    }
    
    // Kroneker Multiplication
    /// Can be used on any matrices
    /// TODO: this can be optimized with block size and changing the order of the interior loops
    Matrix<valType> kroneckerMult( const Matrix<valType>& rhs ){
        Matrix<valType> kroneckerProduct( getRows()*rhs.getRows(), getCols()*rhs.getCols());
        for( size_t j = 0; j < getRows(); j++){
            for( size_t j2 = 0; j2< rhs.getRows(); j2++){
                for( size_t i = 0; i < getCols(); i++){
                    valType v1(operator()(i,j));
                    for( size_t i2 = 0; i2 < rhs.getCols(); i2++){
                        kroneckerProduct(i*rhs.getCols() + i2, j*rhs.getRows() + j2) = v1 * rhs(i2,j2);
                    }
                }
            }
        }
        return kroneckerProduct;
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
    
    /* Matrix Concatenation
     * 0 1  .horzcat( 1 0 ) = 0 1 1 0
     * 0 1            0 1     0 1 0 1
     */
    Matrix<valType> horzcat( const Matrix<valType>& rhs){
#ifdef DBG
        assert(getRows() == rhs.getRows());
#endif
        Matrix<valType> new_matrix(getRows(), getCols()+rhs.getCols());
        for(int i = 0; i < getRows(); i++){
            for( int j = 0; j < getCols(); j++){
                new_matrix(i,j) = operator()(i,j);
            }
            for( int j = 0; j < rhs.getCols(); j++){
                new_matrix( i , j + getCols()) = rhs(i,j);
            }
        }
        return new_matrix;
    }
    
    /* vertical concat
     * 0 1  .vertcat( 1 0 ) = 0 1
     * 0 1            0 1     0 1
     *                        1 0 
     *                        0 1
     */
    
    Matrix<valType> vertcat( const Matrix<valType>& rhs){
#ifdef DBG
        assert(getCols() == rhs.getCols());
#endif
        Matrix<valType> new_matrix(getRows() + rhs.getRows(), getCols());
        for(int j = 0; j < getCols(); j++){
            for( int i = 0; i < getRows(); i++){
                new_matrix(i,j) = operator()(i,j);
            }
            for( int i = 0; i < rhs.getRows(); i++){
                new_matrix( i + getRows(), j) = rhs(i,j);
            }
        }
        return new_matrix;
    }
    
    // print operator
    std::ostream& operator<< (std::ostream & out) {
        for( int i = 0; i < rows; i++){
            for( int j = 0; j < columns; j++){
                out<<operator()(i,j);
            }
            out<<std::endl;
        }
        return out;
    }
private:
    size_t rows, columns;
    std::vector<valType> data;
};


#endif /* LinearAlgebraTools_hpp */
