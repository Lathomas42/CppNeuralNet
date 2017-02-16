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
#include <random>
// this is a debug flag
#define DBG

#ifdef DBG
#include <assert.h>
#endif

/* Matrix class
 *
 * This is a basic resizable matrix class that will come in handy in Neural Networks
 * Probably better to use opencv matrix but this will do for local things
 */
template< typename valType>
class Matrix{
public:

  Matrix():
  rows(0),
  columns(0),
  data(0)
  {}


    // nxm matrix n height m width
  Matrix(size_t nRows, size_t nCol):
  rows(nRows),
  columns(nCol),
  data(nRows*nCol)
  {
      srand((unsigned)time(0));
  }

  // copy constructor


  Matrix( const Matrix<valType>& T):
  rows(T.getRows()),
  columns(T.getCols()),
  data(T.getRows()*T.getCols())
  {
      srand((unsigned)time(0));
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

  template<typename t>
  // copy operator for matrices of different types
  Matrix<valType>& operator=(const Matrix<t>& rhs){
      setSize(rhs.getRows(),rhs.getCols());
      for( int i = 0; i < rows; i++){
          for( int j = 0; j < columns; j++){
              operator()(i,j) = t(rhs(i,j));
          }
      }
  };

  /* --------------------------------------------------------
   * Basic Getters and Setters
   *
   */

  void setSize( size_t nRows, size_t nCol){
      rows = nRows;
      columns = nCol;
      data.resize(nRows*nCol);
  }


  size_t getRows() const{
      return rows;
  }


  size_t getCols() const{
      return columns;
  }


  void fill( valType val ){
      std::fill(data.begin(),data.end(),val);
  }


  void fillRandom( valType valMin, valType valMax ){
  #ifdef DBG
      assert( valMax > valMin);
  #endif
      for( int i = 0; i < getRows(); i++){
          for( int j = 0; j < getCols(); j++){
              double r = rand();
              double r01 = r / RAND_MAX;
              valType x = r01*(valMax - valMin) + valMin;
              operator()(i,j) = x;
          }
      }
  }

  valType max(){
    return *std::max(data.begin(), data.end());
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
      assert(getCols() == rhs.getRows() && i < getRows() && j < rhs.getCols());
  #endif
      valType sum(0);
      for( size_t ind = 0; ind < columns; ind++){
          sum += operator()(i,ind) * rhs(ind,j);
      }
      return sum;
  }

  // row and column vector getters and setters

  Matrix<valType> getRow( size_t i ){
  #ifdef DBG
      assert(i<getRows());
  #endif
      Matrix<valType> row(1,getCols());
      for( int j = 0; j < getCols(); j++){
          row(0,j) = operator()(i,j);
      }
      return row;
  }


  Matrix<valType> getColumn( size_t j ){
  #ifdef DBG
      assert(j<getCols());
  #endif
      Matrix<valType> col(getRows(),1);
      for( int i = 0; i < getRows(); i++){
          col(i,0) = operator()(i,j);
      }
      return col;
  }


  void setRow( size_t i, const Matrix<valType>& row ){
  #ifdef DBG
      assert(row.getCols() == getCols() && row.getRows() == 1 && i < getRows());
  #endif
      for( int j = 0; j < getCols(); j++){
          operator()(i,j) = row(0,j);
      }
  }


  void setColumn( size_t j, const Matrix<valType>& column ){
  #ifdef DBG
      assert(column.getRows() == getRows() && column.getCols() == 1 && j < getCols());
  #endif
      for( int i = 0; i < getRows(); i++){
          operator()(i,j) = column(i,0);
      }
  }

  /*-------------------------------------------
   * Assigment operators
   *
   */

  // replace with multiplication

  Matrix<valType>& operator*=(const valType& mult){
      for( size_t i=0; i < getRows(); i++){
          for(size_t j = 0; j < getCols(); j++){
              operator()(i,j) *= mult;
          }
      }
      return *this;
  }

  // addition assigment

  Matrix<valType>& operator+=(const valType& val){
      for( size_t i=0; i < getRows(); i++){
          for(size_t j = 0; j < getCols(); j++){
              operator()(i,j) += val;
          }
      }
      return *this;
  }

  // division assigment

  Matrix<valType>& operator/=(const valType& val){
      for( size_t i=0; i < getRows(); i++){
          for(size_t j = 0; j < getCols(); j++){
              operator()(i,j) /= val;
          }
      }
      return *this;
  }

  // subtraction assigment

  Matrix<valType>& operator-=(const valType& val){
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

  Matrix<valType> operator*(const valType& mult){
      Matrix<valType> copy(*this);
      copy *= mult;
      return copy;
  }

  // constant division

  Matrix<valType> operator/(const valType& val){
      Matrix<valType> copy(*this);
      copy /= val;
      return copy;
  }

  // constant addition

  Matrix<valType> operator+(const valType& val){
      Matrix<valType> copy(*this);
      copy += val;
      return copy;
  }

  // constant subtraction

  Matrix<valType> operator-(const valType& val){
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
      assert(rhs.getRows() == getCols());
  #endif
      std::cout<<"SIZE"<<getRows()<<","<<rhs.getCols()<<std::endl;
      Matrix<valType> new_matrix(getRows(), rhs.getCols());
      for( size_t i = 0; i < getRows(); i++){
          for( size_t j = 0; j < rhs.getCols(); j++){
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

  // Determinant of a matrix. suprisingly tricky
  valType determinant(){
    if( getCols() != getRows() ){
#ifdef DBG
      assert(false);
#endif
      return -1.0;
    }

    if( rows == 2 && columns == 2 )
      return (operator()(0,0) * operator()(1,1) - operator()(1,0) * operator()(0,1));

    Matrix<valType> submat( getRows() - 1, getCols() - 1);
    int n = rows;
    valType det = 0;
    for( int iter = 0; iter < n; iter++){
      int i2 = 0;
      // skip first row
      for( int i = 1; i < n; i++){
        int j2 = 0;
        for( int j = 0; j < n; j++){
          // skip the iter column
          if( j != iter ){
            submat(i2,j2) = operator()(i,j);
            j2++;
          }
        }
        i2++;
      }
      det += std::pow(-1.0,iter)* operator()(0,iter) * submat.determinant();
    }
    return det;
  }


  // Transpose

  Matrix<valType> T(){
      Matrix<valType> new_matrix(getCols(), getRows());
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
  // Apply a function to the whole matrix and return it as a new matrix

  Matrix<valType> applyFunction( std::function<double (double)> f){
      Matrix<valType> new_matrix(getRows(), getCols());
      for( int i = 0; i < getRows(); i++){
          for( int j = 0; j < getCols(); j++){
              new_matrix(i,j) = f(operator()(i,j));
          }
      }
      return new_matrix;
  }

  valType sumMatrix(){
      valType sum(0.0);
      for( auto i : data){
          sum += i;
      }
      return sum;
  }
  // print operator

  std::ostream& operator<< (std::ostream & out) {
      for( int i = 0; i < rows; i++){
          for( int j = 0; j < columns; j++){
              out<<operator()(i,j)<<" ";
          }
          out<<std::endl;
      }
      return out;
  }

    enum Direction{
        kAll = 0,
        kRows = 1,
        kColumns = 2
    };
private:
    size_t rows, columns;
    std::vector<valType> data;
};


#endif /* LinearAlgebraTools_hpp */
