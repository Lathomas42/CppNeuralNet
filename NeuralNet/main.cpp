//
//  main.cpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/14/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <functional>
#include "LinearAlgebraTools.hpp"
#include "NearestNeighbor.hpp"

double exampleFunc( double x){
    return std::sqrt(x);
}

int main(int argc, const char * argv[]) {
    // insert code here...

    Matrix<double> m2(3,3);
    m2.fill(2.0);
    std::cout<<"m2"<<std::endl;
    m2.operator<<(std::cout);

    Matrix<double> mI(3,3);
    mI.fill(0.0);
    mI(0,0) = 1.0;
    mI(1,1) = 1.0;
    mI(2,2) = 2.0;
    std::cout<<"mI"<<std::endl;
    mI.operator<<(std::cout);


    std::cout<<"Normal Mult"<<std::endl;
    Matrix<double> m3 = m2 * mI;
    m3.operator<<(std::cout);


    std::cout<<"Hadmard Mult"<<std::endl;
    m3 = m2.hadmardMult(mI);
    m3.operator<<(std::cout);

    std::cout<<"Kronker Mult"<<std::endl;
    m3 = m2.kroneckerMult(mI);
    m3.operator<<(std::cout);


    std::cout<<"HorzCat"<<std::endl;
    m3 = m2.horzcat(mI);
    m3.operator<<(std::cout);

    std::cout<<"vertCat"<<std::endl;
    m3 = m2.vertcat(mI);
    m3.operator<<(std::cout);

    std::cout<<"Add"<<std::endl;
    m3 = m2 + mI;
    m3.operator<<(std::cout);

    std::cout<<"Sub"<<std::endl;
    m3 = m2 - mI;
    m3.operator<<(std::cout);

    std::cout<<"take the square root"<<std::endl;
    m3 = m3.applyFunction(exampleFunc);
    m3.operator<<(std::cout);

    std::cout<<"Fill with random between -4 and 4"<<std::endl;
    m3.fillRandom(-4.0, 4.0);
    m3.operator<<(std::cout);

    std::cout<<"Sum of m3"<<std::endl;
    std::cout<<m3.sumMatrix()<<std::endl;

    return 0;
}
