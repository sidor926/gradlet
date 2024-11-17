#include "utils.h"
#include "MLP.h"


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