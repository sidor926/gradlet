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
#include "utils.h"


void test_xor() {
    std::vector<double> ys = {};
    std::vector<std::vector<double>> x = {
        {0, 0}, 
        {0, 1}, 
        {1, 0}, 
        {1, 1}
    };
    std::vector<double> yn = {0, 1, 1, 0}; 

    std::vector<std::shared_ptr<Value>> yn_valobjs = {};
    for (int i = 0; i < yn.size(); i++) {
        yn_valobjs.push_back(std::make_shared<Value>(yn[i]));
    }

    MLP mlp(2, {3, 1}); 

    //creating input vectors with value objects
    std::vector<std::vector<std::shared_ptr<Value>>> xs_valobjs;
    for (const auto& row : x) {
        std::vector<std::shared_ptr<Value>> val_row;
        for (float val : row) {
            val_row.push_back(std::make_shared<Value>(val));
        }
        xs_valobjs.push_back(val_row);
    }

    int epochs = 100;

    int i; 
    for (i = 0; i<epochs; i++) {
        std::vector<std::shared_ptr<Value>> ypred;
        std::vector<std::shared_ptr<Value>> out; 

        for (int j = 0; j < xs_valobjs.size(); j++) { 
            out = mlp(xs_valobjs[j]);
            ypred.push_back(out[0]); //out is a vector of shared_ptr value objects
        }

        auto loss = msee(ypred, yn_valobjs);
        // auto loss = mlp.msee(ypred, yn_valobjs);


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

}

void binary_classification() {
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
}