#include "link_function.hpp"

// TODO: rewrite this functions with dlib only
// i.e. check matrix algebra will work

// These are for a 2D linear regression only

double LinkFunction::likelihood(const column_vector& parameters)
{
   // L(b) = 1/2*||y-Xb||^2
   double result = 0.5 * accu(square(_responses - _predictors * parameters));

   return result;
}

// Evaluate the gradient of the linear regression objective function.
column_vector LinkFunction::gradient(const column_vector& parameters)
{
   // Convert from dlib column matrix to armadillo column matrix
   column_vec gradient_value(parameters.nr());

   // dL(b)/db = X.t()(Xb - y)
   gradient = trans(_predictors) * (_predictors * parameters - _responses);

   return(gradient_value);
}

