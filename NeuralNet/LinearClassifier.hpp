/* Linear Classifier

This basic Linear classifier works on two principles, a score function and a loss
function.

This is the next step up from the Nearest Neighbor class as it does not have to
remember all of the training data.

Score function: maps raw data to a score for each class
Loss function: quantifies how correct a given predicted score is compared to the
expected score

Goal: Minimize loss with respect to the parameters of the score function
*/

/*
Loss Function:

one method for a loss function is to have a delta. Loss only occurs if any other
classes come within delta of the correct class score. If all other classes are
more than delta lower than the correct class's score, the loss is 0.

Problem with this method: the final loss matrix is not unique. there is no
differentiating between W  and aW a>1. To differentiate these, we introduce a
Regulation penalty to discourage large weights
*/
#include "LinearAlgebraTools.hpp"
#include <vector>

class LinearClassifier: BaseClassifier<Matrix<float>>{
public:
  LinearClassifier( int nClasses ){
    delta = 1.0;
  }

  virtual void train(std::vector<Matrix<float>>& input, std::vector<int>& output) override{

  }

  virtual std::vector<int> predict(const std::vector<Matrix<float>>& intput) override{
    return std::vector<int>(0);
  }

  float LossFunction(const Matrix<float>& input, const int correctOutput){
    /*
    - input: a column vector representing one input (eg. an image)
    - correctOutput: the correct classification
    the loss function will return the loss for the set delta
    */
    assert(Weights.getCols() == input.getRows());

    //                      N x D    D x 1
    Matrix<float> scores = Weights * input;
    // get the score for the correct Weight
    float correctScore = scores(correctOutput,0);
    float d = delta;
    // now the loss function
    scores = scores.applyFunction([correctScore,d](double s){return std::max<double>(0.0,s - correctScore + d);});
    return scores.sumMatrix();
  }

  // lambda is a hyperparameter we can optimize with cross validation
  float RegulationPenalty( const float Lambda ){
    Matrix<float> wSquared = Weights.applyFunction([](double x){return x*x;});
    return wSquared.sumMatrix();
  }


private:
  Matrix<float> Weights; // N x D
  Matrix<float> bias; // 1 x N
  float delta;
};
