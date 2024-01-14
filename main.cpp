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
        Layer(int nin, int nout) { 
            for (int i = 0; i < nout; i++) {
                auto neuron = std::make_shared<Neuron>(nin); // initialize nout neurons with nin weights 
                neurons.push_back(neuron);

                // std::cout << "Neuron " << i << ":\n";
                // for (int j = 0; j < nin; ++j) {
                //     std::cout << "  Weight " << j << ": " << neuron->weights[j]->data << "\n";
                // }
                // std::cout << "  Bias: " << neuron->bias->data << "\n";
            }
        }

        void operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
            //outs should be Neuron values calculated with the inputted weights
                for (auto& neuron : neurons) {
                    neuron->operator()(inputs);
                }
        }
};

//Neuron receives the vector x as an input. So neuron should receive size? But Neuron does a dot product inside it? 
//Layer just receives input size

static std::random_device rd; 
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> distr(-1, 1);


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

int main() {
    auto x = std::vector<std::shared_ptr<Value>>(); //vector of pointers to a Value object
    
    //make some xi's
    for (int i = 0; i < 3; i++) {
        std::shared_ptr<Value> xi = std::make_shared<Value>(distr(gen)); 
        x.push_back(xi);
        std::cout << "x" << i << ": " << x[i]->data << std::endl;
    }

    // auto n1 = std::make_shared<Neuron>(x.size());
    // (*n1)(x);
    // std::cout << "n1 out " << n1 << ": " << n1->out->data << std::endl;

    // auto n1 = Neuron(3);
    // auto our_neuron = n1(x);

    // std::cout << "n1 out " << &n1 << ": " << n1.out->data << std::endl;

    // Create a Layer instance
    Layer layer(3, 2); // Assuming 3 inputs and 2 neurons in the layer

    // Pass inputs to the layer and get the outputs
    layer(x);
    printLayerDetails(layer);

    // Print outputs
    return 0;
}


/* 
Manually tested backprop for the following expressions: 

e = a*b + d 
e = a*b*d
e = a*b + a*b 
g = (a*b+d)*f 

*/


