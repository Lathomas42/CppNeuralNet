//
//  main.cpp
//  NeuralNet
//
//  Created by Logan Thomas on 1/14/17.
//  Copyright © 2017 Logan Thomas. All rights reserved.
//

#include <iostream>
#include "LinearAlgebraTools.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
    
    Matrix<double> m(3,3);
    m.fill(2.0);
    
    m.operator<<(std::cout);
    
    return 0;
}
