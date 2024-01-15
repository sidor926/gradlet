// #pragma once

#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <random>


class Value: public std::enable_shared_from_this<Value>{ 
    public: 
        float data; 
        float grad;
        std::function<void()> _backward;
        std::set<std::shared_ptr<Value>> _parents; 

    public: 
        Value(float data, std::set<std::shared_ptr<Value>> _parents = {}) { 
            this->data = data;
            this->grad = 0.0;
            this->_backward = nullptr;
            this->_parents = _parents ; 
        }

        bool operator<(const Value& other) const {
            return data < other.data;
        }

        std::shared_ptr<Value> add(std::shared_ptr<Value> other) {
            auto out = std::make_shared<Value>(this->data + other->data);
            
            out->_parents.insert(shared_from_this());
            out->_parents.insert(other);

            out->_backward = [this, other, out]() {
                this->grad += 1.0 * out->grad; 
                other->grad += 1.0 * out->grad; 
            };

            return out; 
        }

        std::shared_ptr<Value> mul(std::shared_ptr<Value> other) {
            auto out = std::make_shared<Value>(this->data * other->data); 

            out->_parents.insert(shared_from_this());
            out->_parents.insert(other);

            out->_backward = [this, other, out]() {
                this->grad += other->data * out->grad; 
                other->grad += this->data * out->grad; 
            };

            return out; 
    }
};

void build_topo_order(const std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited) {
    if (visited.find(v) == visited.end()) { //if v not in visited
        visited.insert(v);

        for (const auto parent : v->_parents) { //won't enter the loop for orphan nodes
            build_topo_order(parent, topo, visited); 
        }
        topo.push_back(v);
    }
}

void backward(std::shared_ptr<Value> root) {
    std::vector<std::shared_ptr<Value>> topo; 
    std::set<std::shared_ptr<Value>> visited; 

    build_topo_order(root, topo, visited);

    root->grad = 1.0; 

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) { //traverse in reverse order innit
        if ((*it)->_backward) { // input nodes don't have a _backward
            (*it)->_backward();
        }
    }
}


/*
self->grad += 1.0 * out->grad;
other->grad += 1.0 * out->grad;

 def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
*/

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
    std::cout << "Printing layer details:" << "\n";
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

        void operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            std::vector<std::shared_ptr<Value>> inps = inputs;
            for (int i = 0; i < mlp_layers.size(); i++) { //complete feedforward
                // layer(inputs);
                // printLayerDetails(*mlp_layers[i]);
                inps = mlp_layers[i]->operator()(inps);
                printLayerDetails(*mlp_layers[i]);
            }
        }
};

//Layer just receives input size

static std::random_device rd; 
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> distr(-1, 1);

int main() {
    auto x = std::vector<std::shared_ptr<Value>>(); //vector of pointers to a Value object
    
    //make some xi's
    for (int i = 0; i < 3; i++) {
        std::shared_ptr<Value> xi = std::make_shared<Value>(distr(gen)); 
        x.push_back(xi);
        std::cout << "x" << i << ": " << x[i]->data << std::endl;
    }

    auto mlp_definition = std::vector<int>{3, 2};
    MLP mlp(x.size(), mlp_definition); //constructor takes the input feature size 
    mlp(x); 

    // Print outputs
    return 0;
};


/* 
Manually tested backprop for the following expressions: 

e = a*b + d 
e = a*b*d
e = a*b + a*b 
g = (a*b+d)*f 

*/


