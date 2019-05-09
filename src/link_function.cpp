#include "link_function.hpp"

// These are for a 2D linear regression only

double LinearLink::likelihood(const column_vector& parameters)
{
   // L(b) = 1/2*||y-Xb||^2
   double result = 0.5 * length_squared(_responses - _predictors * parameters);

   return result;
}

// Evaluate the gradient of the linear regression objective function.
column_vector LinearLink::gradient(const column_vector& parameters)
{
   // Convert from dlib column matrix to armadillo column matrix
   column_vec gradient_value(parameters.nr());

   // dL(b)/db = X.t()(Xb - y)
   gradient = trans(_predictors) * (_predictors * parameters - _responses);

   return(gradient_value);
}

