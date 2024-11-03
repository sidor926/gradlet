#pragma once

#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <cmath>

class Value: public std::enable_shared_from_this<Value> {
    public:
        float data;
        float grad;
        std::function<void()> _backward;
        std::set<std::shared_ptr<Value>> _parents;

        //this line 
        Value(float data, std::set<std::shared_ptr<Value>> _parents = {});

        bool operator<(const Value& other) const;

        std::shared_ptr<Value> add(std::shared_ptr<Value> other);
        std::shared_ptr<Value> mul(std::shared_ptr<Value> other);
        std::shared_ptr<Value> pow(float other);
        std::shared_ptr<Value> tanh();
        std::shared_ptr<Value> exp();
        std::shared_ptr<Value> sub(std::shared_ptr<Value> other);
        std::shared_ptr<Value> divide(std::shared_ptr<Value> other);
        std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
        std::shared_ptr<Value> operator-();
        std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
        void printGraph(int depth) const;
};  

void build_topo_order(const std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited);
void backward(std::shared_ptr<Value> root);
