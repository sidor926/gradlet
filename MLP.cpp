#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <random>
#include <cmath>
#include "Value.h"
#include "MLP.h"
// #include "tests.cpp"


Neuron::Neuron(int inp_size) {
    static std::random_device rd; 
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> distr(-1, 1);

    for (int i = 0; i < inp_size; i++) {
        std::shared_ptr<Value> weight = std::make_shared<Value>(distr(gen));
        weights.push_back(weight);
    }
    bias = std::make_shared<Value>(distr(gen));
    // std::cout << "Bias: " << bias->data << std::endl;

    // for (int i = 0; i < inp_size; ++i) {
    //     std::cout << "w" << i << ": " << weights[i]->data << std::endl;
    // }
    out = std::make_shared<Value>(0);  
}

//TODO: add check to see if input dim == num weights, right now just using weight.size 
std::shared_ptr<Value> Neuron::operator() (const std::vector<std::shared_ptr<Value>>& inputs) {
    out = std::make_shared<Value>(0);  //this was the problem
    for (int j = 0; j < weights.size(); j++) {
        // std::cout << "input: " << inputs[j]->data << "\n";
        // std::cout << "weight: " << weights[j]->data << "\n";
        out = out->add(inputs[j]->mul(weights[j]));
    }

    out = out->add(bias);
    // std::cout << "out data: " << out->data << "\n";
    return out;
}

Layer::Layer(int inp_size, int num_neurons) {
    for (int i = 0; i < num_neurons; i++) {
        auto neuron = std::make_shared<Neuron>(inp_size);
        neurons.push_back(neuron);
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    //outs should be Neuron values calculated with the inputted weights
    auto outs = std::vector<std::shared_ptr<Value>>(); 
        for (auto& neuron : neurons) {
        auto out = neuron->operator()(inputs);
        outs.push_back(out);
    }
    return outs; 
}


MLP::MLP(int inp_size, std::vector<int> layer_sizes) {
    std::shared_ptr<Layer> l; 
    for (int i = 0; i <layer_sizes.size(); i++) {
        if (i == 0) {
            l = std::make_shared<Layer>(inp_size, layer_sizes[i]); 
        } else { 
            l = std::make_shared<Layer>(layer_sizes[i-1], layer_sizes[i]); 
        }
        mlp_layers.push_back(l);
        // printLayerDetails(*l);
    }
    // std::cout << "Created all layers inside mlp_layers" << "\n";
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    std::vector<std::shared_ptr<Value>> inps = inputs;
    for (int i = 0; i < mlp_layers.size(); i++) { //complete feedforward
        // layer(inputs);
        // printLayerDetails(*mlp_layers[i]);
        inps = mlp_layers[i]->operator()(inps);
        // std::cout << "Printing layer " << i << " details:" << "\n";
        // printLayerDetails(*mlp_layers[i]);
    }
    return inps;
}

std::vector<std::shared_ptr<Value>> MLP::get_all_params() {
    std::vector<std::shared_ptr<Value>> parameters;
    for (int i = 0; i < mlp_layers.size(); i++) {
        for (int j = 0; j < mlp_layers[i]->neurons.size(); j++) {
            for (int k = 0; k < mlp_layers[i]->neurons[j]->weights.size(); k++) {
                //add all weights to the array
                parameters.push_back(mlp_layers[i]->neurons[j]->weights[k]);
            }
            parameters.push_back(mlp_layers[i]->neurons[j]->bias); //also a shared pointer 
        // break; 
        }
    // break;
    }
    return parameters;
}
