#include <iostream>
#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <random>
#include <cmath>
#include "Value.h"


void test_sigmoid() {

    std::cout << "Testing Forward.." << "\n";
    std::vector<int> inputs = {1, 2, 3, -1, -2};
    std::vector<double> inputs_sigmoid_true = {0.7310585786, 0.880797078, 0.9525741268, 0.2689414214, 0.119202922};
    std::vector<std::shared_ptr<Value>> inputs_sigmoid_calculated = {}; 
    std::vector<double> inputs_grads = {0.1966,0.1050,0.0452,0.1966,0.1050};
    std::vector<std::shared_ptr<Value>> inputs_valObjs = {};
    
    for (auto& inp : inputs) {
        inputs_valObjs.push_back(std::make_shared<Value>(inp)); 
    }

    for (auto& inp : inputs_valObjs) {
        //call sigmoid on all input value objects
        inputs_sigmoid_calculated.push_back(inp->sigmoid());
    }

    bool correct = true; 
    //create vector containing incorrect value
    for (int i = 0; i < inputs.size(); i++) {
        if (std::abs(inputs_sigmoid_true[i] - inputs_sigmoid_calculated[i]->data) > 1e-6) {
            correct = false; 
            std::cout << inputs_sigmoid_true[i];
            //add to vector containing incorrect value
        } 
    }

    if (!correct) {
        std::cout << "Sigmoid calculated incorrectly" << "\n"; 
    } else { std:: cout << "Sigmoid calculated correctly for input vector" << "\n";}
 
    std::cout << "Testing Backward.." << "\n";

    for (auto& sigmoid_outs : inputs_sigmoid_calculated) {
        sigmoid_outs->grad = 1.0;
        sigmoid_outs->_backward(); 
    }

    bool grads_correct = true;
    for (int j = 0; j < inputs.size(); j++) {
         if (std::abs(inputs_valObjs[j]->grad - inputs_grads[j]) > 1e-4) {
            grads_correct = false; 
            std::cout << inputs_valObjs[j]->grad << "\n";
            //add to vector containing incorrect value
        } 
    }

    if (!grads_correct) {
        std::cout << "Sigmoid grads calculated incorrectly" << "\n"; 
    } else { std:: cout << "Sigmoid calculated correctly for input vector" << "\n";}
}

/* 

Manually tested backprop for the following expressions: 

e = a*b + d 
e = a*b*d
e = a*b + a*b 
g = (a*b+d)*f 

*/

