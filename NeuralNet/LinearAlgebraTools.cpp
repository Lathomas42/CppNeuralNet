//
//  LinearAlgebraTools.cpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/15/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#include "LinearAlgebraTools.hpp"

template<typename valType>
Matrix<valType>::Matrix():
rows(0),
columns(0),
data(0)
{}


template<typename valType>
Matrix<valType>::Matrix(size_t nRows, size_t nCol):
rows(nRows),
columns(nCol),
data(nRows*nCol)
{
}

// copy constructor

template<typename valType>
Matrix<valType>::Matrix( const Matrix<valType>& T):
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
template<typename valType>
Matrix<valType>& Matrix<valType>::operator=(const Matrix<valType>& right){
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
template<typename valType>
void Matrix<valType>::setSize( size_t nRows, size_t nCol){
    rows = nRows;
    columns = nCol;
    data.resize(nRows*nCol);
}

template<typename valType>
size_t Matrix<valType>::getRows() const{
    return rows;
}

template<typename valType>
size_t Matrix<valType>::getCols() const{
    return columns;
}

template<typename valType>
void Matrix<valType>::fill( valType val ){
    std::fill(data.begin(),data.end(),val);
}

// element
template<typename valType>
valType& Matrix<valType>::operator()(size_t i, size_t j){
#ifdef DBG
    assert(i < rows && j < columns);
#endif
    return data[i*columns + j];
}

template<typename valType>
valType Matrix<valType>::operator()(size_t i, size_t j) const{
#ifdef DBG
    assert(i < rows && j < columns);
#endif
    return data[i*columns + j];
}

template<typename valType>
valType Matrix<valType>::getMultVal( const Matrix<valType>& rhs, size_t i, size_t j){
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
template<typename valType>
Matrix<valType>& Matrix<valType>::operator*=(const valType& mult){
    for( size_t i=0; i < getRows(); i++){
        for(size_t j = 0; j < getCols(); j++){
            operator()(i,j) *= mult;
        }
    }
    return *this;
}

// addition assigment
template<typename valType>
Matrix<valType>& Matrix<valType>::operator+=(const valType& val){
    for( size_t i=0; i < getRows(); i++){
        for(size_t j = 0; j < getCols(); j++){
            operator()(i,j) += val;
        }
    }
    return *this;
}

// division assigment
template<typename valType>
Matrix<valType>& Matrix<valType>::operator/=(const valType& val){
    for( size_t i=0; i < getRows(); i++){
        for(size_t j = 0; j < getCols(); j++){
            operator()(i,j) /= val;
        }
    }
    return *this;
}

// subtraction assigment
template<typename valType>
Matrix<valType>& Matrix<valType>::operator-=(const valType& val){
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
template<typename valType>
Matrix<valType> Matrix<valType>::operator*(const valType& mult){
    Matrix<valType> copy(*this);
    copy *= mult;
    return copy;
}

// constant division
template<typename valType>
Matrix<valType> Matrix<valType>::operator/(const valType& val){
    Matrix<valType> copy(*this);
    copy /= val;
    return copy;
}

// constant addition
template<typename valType>
Matrix<valType> Matrix<valType>::operator+(const valType& val){
    Matrix<valType> copy(*this);
    copy += val;
    return copy;
}

// constant subtraction
template<typename valType>
Matrix<valType> Matrix<valType>::operator-(const valType& val){
    Matrix<valType> copy(*this);
    copy -= val;
    return copy;
}


/* ------------------------------------------
 * Matrix Assignment Operators
 *
 */

// basic matrix addition assigment operator
template<typename valType>
Matrix<valType>& Matrix<valType>::operator+=(const Matrix<valType>& rhs){
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
template<typename valType>
Matrix<valType>& Matrix<valType>::operator-=(const Matrix<valType>& rhs){
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
template<typename valType>
Matrix<valType> Matrix<valType>::operator+(const Matrix<valType>& rhs){
#ifdef DBG
    assert(getRows() == rhs.getRows() && getCols() == rhs.getCols());
#endif
    Matrix<valType> copy(*this);
    copy += rhs;
    return copy;
}

// basic matrix subtraction
template<typename valType>
Matrix<valType> Matrix<valType>::operator-(const Matrix<valType>& rhs){
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
template<typename valType>
Matrix<valType> Matrix<valType>::operator*( const Matrix<valType>& rhs){
#ifdef DBG
    assert(rhs.getRows() == getCols());
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
template<typename valType>
Matrix<valType> Matrix<valType>::hadmardMult( const Matrix<valType>& rhs){
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
template<typename valType>
Matrix<valType> Matrix<valType>::kroneckerMult( const Matrix<valType>& rhs ){
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
template<typename valType>
Matrix<valType> Matrix<valType>::T(){
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
template<typename valType>
Matrix<valType> Matrix<valType>::horzcat( const Matrix<valType>& rhs){
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
template<typename valType>
Matrix<valType> Matrix<valType>::vertcat( const Matrix<valType>& rhs){
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
template<typename valType>
std::ostream& Matrix<valType>::operator<< (std::ostream & out) {
    for( int i = 0; i < rows; i++){
        for( int j = 0; j < columns; j++){
            out<<operator()(i,j);
        }
        out<<std::endl;
    }
    return out;
}

// specific instantiations
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;




