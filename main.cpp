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

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) { // input nodes don't have a _backward
            (*it)->_backward();
        }
    }
}

//how are we handling the bakward cases when there is no leaf node? 


/*
self->grad += 1.0 * out->grad;
other->grad += 1.0 * out->grad;

 def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
*/

class Neuron {
    private: 
        std::random_device rd; 

    public: 
        std::vector<std::shared_ptr<Value>> weights;
        std::vector<std::shared_ptr<Value>> biases;
        
    Neuron(int no_of_neurons) {
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distr(-1, 1);

        for (int i = 0; i < no_of_neurons; ++i) {
            // Initialize each weight with a random value
            auto weight = std::make_shared<Value>(distr(gen));
            weights.push_back(weight);
        }

        for (int i = 0; i < no_of_neurons; ++i) {
            //print all data values of each Value node
        }

        // Initialize bias with a random value
        auto bias = std::make_shared<Value>(distr(gen));
    }
};


int main() {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(3); 
    auto c = a->mul(b);

    //using the same node twice bug needs to solved for, maybe its already solved. that is up next. 
    auto d = std::make_shared<Value>(4);
    auto e = c->add(d);

    auto f = std::make_shared<Value>(5);
    auto g = f->mul(e);

    // e->grad = 1.0; 
    // e->_backward(); //should save gradients for c and d

    backward(g);
    std::cout << "g data: " << g->data << std::endl;

    std::cout << "f data: " << f->data << std::endl;
    std::cout << "f grad: " << f->grad << std::endl;

    std::cout << "e data: " << e->data << std::endl;
    std::cout << "e grad: " << e->grad <<std::endl;
    
    // c->_backward(); //should save gradients for a and b. 
    std::cout << "d data: " << d->data << std::endl;
    std::cout << "d grad: " << d->grad << std::endl;
    std::cout << "c data: " << c->data << std::endl;
    std::cout << "c grad: " << c->grad << std::endl;

    std::cout << "b data: " << b->data << std::endl;
    std::cout << "b grad: " << b->grad << std::endl;

    std::cout << "a data: " << a->data << std::endl;
    std::cout << "a grad: " << a->grad << std::endl;


    // std::cout << &c << std::endl;
    // for (const auto& parent : e->_parents) {
    //     std::cout << "Parent data: " << parent->data << std::endl;
    // }

    return 0;
}



/* 
Manually tested backprop for the following expressions: 

e = a*b + d 
e = a*b*d
e = a*b + a*b 
g = (a*b+d)*f 

*/


