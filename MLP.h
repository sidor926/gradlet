#ifndef MLP_H
#define MLP_H

#include <vector>
#include <memory>
#include "Value.h"

class Neuron {
public:
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;
    std::shared_ptr<Value> out;

    Neuron(int inp_size);
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
};

class Layer {
public:
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer(int inp_size, int num_neurons);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
};

// void printLayerDetails(Layer& layer);

class MLP {
public:
    std::vector<std::shared_ptr<Layer>> mlp_layers;

    MLP(int inp_size, std::vector<int> layer_sizes);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);
    std::vector<std::shared_ptr<Value>> get_all_params();
};


#endif
