#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <memory>
#include "Value.h"
#include "MLP.h"

// Mean Squared Error (MSE) function
std::shared_ptr<Value> msee(std::vector<std::shared_ptr<Value>> ypred, std::vector<std::shared_ptr<Value>> ys);
void printLayerDetails(Layer& layer);
#endif

