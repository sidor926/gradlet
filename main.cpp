// #pragma once

#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <random>
#include <cmath>
#include "Value.h"


class Neuron {    
    public: 
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias;
        std::shared_ptr<Value> out;
        
    Neuron(int inp_size) {
        static std::random_device rd; 
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> distr(-1, 1);
        
        //initialize random weights
        for (int i = 0; i < inp_size; ++i) {
            std::shared_ptr<Value> weight = std::make_shared<Value>(distr(gen));
            weights.push_back(weight);
        }

        //initialize random bias 
        bias = std::make_shared<Value>(distr(gen));
        // std::cout << "Bias: " << bias->data << std::endl;

        // for (int i = 0; i < inp_size; ++i) {
        //     std::cout << "w" << i << ": " << weights[i]->data << std::endl;
        // }
    }

    //calling the neuron object
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        //dot product
        double outData = 0.0; 
        for (int j = 0; j < inputs.size(); ++j) {
            outData += weights[j]->data * inputs[j]->data;
        }
        outData += bias->data;
        out = std::make_shared<Value>(outData);   

        return out;
    }
};

class Layer {
    public: 
        std::vector<std::shared_ptr<Neuron>> neurons; //Layer is vector that holds pointers to Neuron objects. public for now.

    public: 
        Layer(int num_input_weights, int num_neurons) { 
            for (int i = 0; i < num_neurons; i++) {
                auto neuron = std::make_shared<Neuron>(num_input_weights); 
                neurons.push_back(neuron);
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            //outs should be Neuron values calculated with the inputted weights
            auto outs = std::vector<std::shared_ptr<Value>>();
            for (auto& neuron : neurons) {
                auto out = neuron->operator()(inputs);
                outs.push_back(out);
            }

            return outs; 
        }
};

void printLayerDetails(Layer &layer) {
    for (int i = 0; i < layer.neurons.size(); ++i) {
        auto& neuron = layer.neurons[i];
        std::cout << "Neuron " << i << ":\n";
        for (int j = 0; j < neuron->weights.size(); ++j) {
            std::cout << "  Weight " << j << ": " << neuron->weights[j]->data << "\n";
        }
        std::cout << "  Bias: " << neuron->bias->data << "\n";
        std::cout << "  Output: " << neuron->out->data << "\n";
    }
}

class MLP { 
    public:
        std::vector<std::shared_ptr<Layer>> mlp_layers; //vector of pointers to layer objects
    
    public: 
        MLP(int input_vector_size, std::vector<int> layer_sizes) {
            std::shared_ptr<Layer> l;
            for (int i = 0; i < layer_sizes.size(); i++) {
                if (i == 0) { //first hidden layer 
                    l = std::make_shared<Layer>(input_vector_size, layer_sizes[i]); //num weights per neuron, num_neurons
                } else {
                    l = std::make_shared<Layer>(layer_sizes[i-1], layer_sizes[i]); 
                }
                mlp_layers.push_back(l);
                // printLayerDetails(*l);
            }
            // std::cout << "Created all layers inside mlp_layers" << "\n";
        }

        std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
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
};

//Layer just receives input size

std::shared_ptr<Value> msee(std::vector<std::shared_ptr<Value>> ypred, std::vector<float> ys) {
    //expected output - output of last neuron)^2)
    std::shared_ptr<Value> error = std::make_shared<Value>(0.0);

    for (int i = 0; i < ypred.size(); i++) {
        error->data += pow(ys[i] - ypred[i]->data, 2); 
        // std::cout << "Error " << i << ": " << error << "\n";
    }
    
    error->data = error->data / ypred.size();
    // std::cout << "Error: " << error->data << "\n";
    return error; 
}


static std::random_device rd; 
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> distr(-1, 1);

int main() {
    auto x = std::vector<std::shared_ptr<Value>>(); //vector of pointers to a Value object
    
    //make some xi's
    // for (int i = 0; i < 3; i++) {
    //     std::shared_ptr<Value> xi = std::make_shared<Value>(distr(gen)); 
    //     x.push_back(xi);
    //     std::cout << "x" << i << ": " << x[i]->data << std::endl;
    // }

    // auto mlp_definition = std::vector<int>{3, 2};
    // MLP mlp(x.size(), mlp_definition); //constructor takes the input feature size 
    // mlp(x); 

    // std::vector<std::shared_ptr<Value>> input_values;
    // for (auto& val : {2.0, 3.0, -1.0}) {
    //     input_values.push_back(std::make_shared<Value>(val));
    // }

    std::vector<int> layer_sizes = {4, 4, 1}; 
    

    // Forward pass
    // mlp(input_values);
    std::vector<std::vector<float>> xs = { //what is xs and ys? 
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    // std::cout << "Printing values " << xs[0][0] << " details:" << "\n";
    std::vector<float> ys = {1.0, -1.0, -1.0, 1.0};

    MLP mlp(3, layer_sizes);
    std::vector<std::shared_ptr<Value>> input_values;
    std::vector<std::shared_ptr<Value>> ypred;
    for (int i = 0; i < 4; i++) { //all input vectors
        for (auto& val : xs[i]) { //each value in the input vector
            // std::cout << "Each val " << val << " details" << "\n";
            input_values.push_back(std::make_shared<Value>(val));
        }
        auto out = mlp(input_values); //forward pass 
        ypred.push_back(out[0]);
        input_values.clear(); 
        // break;
    }

    //print ypred 
    // for (int i = 0; i < ypred.size(); i++) {
    //     std::cout << "ypred " << i << ": " << ypred[i]->data << "\n";
    // }

    auto loss = msee(ypred, ys);
    std::cout << "Parents of loss:\n" << loss->_parents.size() << "\n";

    // for (const auto& parent : loss->_parents) {
    //     if (parent) {  // Check if the parent pointer is not null
    //         std::cout << "Parent data: " << parent->data << ", grad: " << parent->grad << "\n";
    //     } else {
    //         std::cout << "Null parent\n";
    //     }
    // }


    // loss

    //print the type of loss 
    // std::cout << "Type of loss: " << typeid(loss).name() << "\n";
    // std::cout << mlp.mlp_layers[0]->neurons[0]->weights[0]->grad << "\n";
    // backward(loss); //backprop
    // std::cout << mlp.mlp_layers[0]->neurons[0]->weights[0]->grad << "\n"; //should be 0.0

    return 0;
};


/* 
Manually tested backprop for the following expressions: 

e = a*b + d 
e = a*b*d
e = a*b + a*b 
g = (a*b+d)*f 

g++ -std=c++14 -c Value.cpp -o Value.o && g++ -std=c++14 -c main.cpp -o main.o && g++ -std=c++14 Value.o main.o -o main && ./main



*/


