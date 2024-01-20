#include "Value.h"

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

        std::shared_ptr<Value> pow(float other) {
            auto out = std::make_shared<Value>(std::pow(this->data, other));

            out->_parents.insert(shared_from_this());

            out->_backward = [this, other, out]() {
                this->grad += other * std::pow(this->data, other - 1) * out->grad;
            };

            return out;
    }

        std::shared_ptr<Value> tanh() {
            double x = this->data;
            double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
            auto out = std::make_shared<Value>(t);

            out->_parents.insert(shared_from_this());

            out->_backward = [this, t, out]() {
                this->grad += (1 - t * t) * out->grad;
        };

        return out;
    }

        std::shared_ptr<Value> exp() {
            double x = this->data;
            auto out = std::make_shared<Value>(std::exp(x));

            out->_parents.insert(shared_from_this());

            out->_backward = [this, out]() {
                this->grad += out->data * out->grad; // Gradient of exp(x) w.r.t x is exp(x)
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
