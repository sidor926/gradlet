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

        Value(float data, std::set<std::shared_ptr<Value>> _parents = {});

        bool operator<(const Value& other) const;

        std::shared_ptr<Value> add(std::shared_ptr<Value> other);
        std::shared_ptr<Value> mul(std::shared_ptr<Value> other);
        std::shared_ptr<Value> pow(float other);
        std::shared_ptr<Value> tanh();
        std::shared_ptr<Value> exp();
};

void build_topo_order(const std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited);
void backward(std::shared_ptr<Value> root);
