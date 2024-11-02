#include "Value.h"
#include <vector>  // Include this for std::vector

// Implementations of Value methods
Value::Value(float data, std::set<std::shared_ptr<Value>> _parents) {
    // Constructor implementation
        this->data = data;
        this->grad = 0.0;
        this->_backward = nullptr;
        this->_parents = _parents ; 

}

bool Value::operator<(const Value& other) const {
    // Implementation
    return data < other.data;
}

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data + other->data);
    
    out->_parents.insert(shared_from_this());
    out->_parents.insert(other);

    out->_backward = [this, other, out]() {
        this->grad += 1.0 * out->grad; 
        other->grad += 1.0 * out->grad; 
    };

    return out; 
}

std::shared_ptr<Value> Value::mul(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data * other->data); 

    out->_parents.insert(shared_from_this());
    out->_parents.insert(other);

    out->_backward = [this, other, out]() {
        this->grad += other->data * out->grad; 
        other->grad += this->data * out->grad; 
    };

    return out; 
}

std::shared_ptr<Value> Value::pow(float other) {
    auto out = std::make_shared<Value>(std::pow(this->data, other));

    out->_parents.insert(shared_from_this());

    out->_backward = [this, other, out]() {
        this->grad += other * std::pow(this->data, other - 1) * out->grad;
    };

    return out;
}

std::shared_ptr<Value> Value::tanh() {
    double x = this->data;
    double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    auto out = std::make_shared<Value>(t);

    out->_parents.insert(shared_from_this());

    out->_backward = [this, t, out]() {
        this->grad += (1 - t * t) * out->grad;
};

return out;
}

std::shared_ptr<Value> Value::exp() {
    double x = this->data;
    auto out = std::make_shared<Value>(std::exp(x));

    out->_parents.insert(shared_from_this());

    out->_backward = [this, out]() {
        this->grad += out->data * out->grad; // Gradient of exp(x) w.r.t x is exp(x)
    };

    return out;
}


std::shared_ptr<Value> Value::operator/(std::shared_ptr<Value> other) {
    return this->mul(other->pow(-1));
}

// __neg__: -self
std::shared_ptr<Value> Value::operator-() {
    return this->mul(std::make_shared<Value>(-1));
}

// __sub__: self - other
std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value> other) {
    return this->add(-(*other));
}


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

void Value::printGraph(int depth) const {
    // Print indentation for visual hierarchy
    for (int i = 0; i < depth; ++i) {
        std::cout << "  ";
    }
    // Print current node's data and gradient
    std::cout << "Node data: " << data << ", grad: " << grad << "\n";

    // Recursively print all parents (ancestors in the graph)
    for (const auto& parent : _parents) {
        if (parent) {  // Ensure the parent pointer is not null
            parent->printGraph(depth + 1);
        }
    }
}

