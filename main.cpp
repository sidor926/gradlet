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
        std::shared_ptr<Value> out = std::make_shared<Value>(0);  
        
    Neuron(int inp_size) {
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
    }

    //TODO: add check to see if input dim == num weights, right now just using weight.size 
    std::shared_ptr<Value> operator() (const std::vector<std::shared_ptr<Value>>& inputs) {
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

    std::vector<std::shared_ptr<Value>> parameters; 

    
};


class Layer {
    public: 
        std::vector<std::shared_ptr<Neuron>> neurons; 
    
    Layer(int inp_size, int num_neurons) {
        for (int i = 0; i < num_neurons; i++) {
            auto neuron = std::make_shared<Neuron>(inp_size);
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
};

class MLP { 
    public: 
        std::vector<std::shared_ptr<Layer>> mlp_layers; 
        std::vector<std::shared_ptr<Value>> parameters; 

    public: 
        MLP(int inp_size, std::vector<int> layer_sizes) {
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

        std::vector<std::shared_ptr<Value>> get_all_params() {
            parameters.clear();
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



        };

std::shared_ptr<Value> msee(std::vector<std::shared_ptr<Value>> ypred, std::vector<std::shared_ptr<Value>> ys) {
    //expected output - output of last neuron)^2)
    std::shared_ptr<Value> error = std::make_shared<Value>(0.0);

    for (int i = 0; i < ypred.size(); i++) {
        // std::cout << ys[i]->data << "\n";
        error = error->add(ys[i]->sub(ypred[i])->pow(2));
        // std::cout << "Error " << i << ": " << error << "\n";
    }
    
    error = error->divide(std::make_shared<Value>(ypred.size()));
    // std::cout << "Error: " << error->data << "\n";
    return error; 
}

int main() {
    auto x = std::vector<std::shared_ptr<Value>>(); //vector of pointers to a Value object
    std::vector<int> layer_sizes = {4, 4, 1}; 
    
    std::vector<std::vector<float>> xs = { 
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };


    std::vector<float> ys = {1.0, -1.0, -1.0, 1.0};
    std::vector<std::shared_ptr<Value>> ys_valobjs; 
    for (int i = 0; i < ys.size(); i++) {
        ys_valobjs.push_back(std::make_shared<Value>(ys[i]));
    }


    MLP mlp(3, layer_sizes);

    std::vector<std::vector<std::shared_ptr<Value>>> xs_valobjs;
    for (const auto& row : xs) {
        std::vector<std::shared_ptr<Value>> val_row;
        for (float val : row) {
            val_row.push_back(std::make_shared<Value>(val));
        }
        xs_valobjs.push_back(val_row);
    }

    int epochs = 100;
    for (int i = 0; i < epochs; i++) {

        std::vector<std::shared_ptr<Value>> ypred;
        std::vector<std::shared_ptr<Value>> out; 

        for (int j = 0; j < xs_valobjs.size(); j++) {
            out = mlp(xs_valobjs[j]);
            ypred.push_back(out[0]);
        }

        auto loss = msee(ypred, ys_valobjs);

        auto all_params = mlp.get_all_params();
        for (int k = 0; k < all_params.size(); k++) {
            all_params[k]->grad = 0.0; 
        }
        backward(loss);

        for (int l = 0; l < all_params.size(); l++) {
            all_params[l]->data += -0.01 * all_params[l]->grad; 
        } 

        std::cout << "Iteration: " << i << " Loss: " << loss->data << "\n";
    
    }

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


