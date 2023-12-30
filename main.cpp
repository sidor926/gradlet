#include <iostream>
#include <functional>
#include <memory>
#include <set>



class Value: public std::enable_shared_from_this<Value>{ 
    public: 
        float data; 
        float grad;
        std::function<void()> _backward;
        std::set<std::shared_ptr<Value>> _parents; //a set that contains pointers to the nodes that created it

    public: 

        Value(float data, std::set<std::shared_ptr<Value>> _parents = {}) { 
            this->data = data;
            this->grad = 0.0;
            this->_backward = nullptr;
            this->_parents = _parents ; //why this line?
        }

        bool operator<(const Value& other) const {
            return data < other.data;
        }

        std::shared_ptr<Value> add(std::shared_ptr<Value> other) {
            auto out = std::make_shared<Value>(this->data + other->data);
            
            out->_parents.insert(shared_from_this());
            out->_parents.insert(other);



            return out; 
        }

    }    
;


/*
self->grad += 1.0 * out->grad;
other->grad += 1.0 * out->grad;
*/

int main() {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(3); 
    auto c = a->add(b);

    // std::cout << a << std::endl;
    // std::cout << b <<std::endl;


    // std::cout << &c << std::endl;
    // Example usage
    for (const auto& parent : c->_parents) {
        std::cout << "Parent data: " << parent->data << std::endl;
    }

    
    return 0;
}


