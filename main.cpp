#include <iostream>
#include <set>


class Value { 
    public: 
        float data; 
        float grad;
        std::function<void()> _backward;
        std::set<Value*> _parents; //a set that contains pointers to the nodes that created it

    public: 
        Value(float data, std::set<Value*> _parents = {}) { 
            this->data = data;
            this->grad = 0.0;
            this->_backward = nullptr;
            this->_parents = std::set<Value*>(); //ptr to an empty set that will have the parents. 
        }

        Value add(Value *other) { //using references and not pointers
            float out_data = this->data + other->data; 
            Value out = Value(out_data, {this, other});  

            auto _backward = [this, other, out]() { //my problem with this is that its inconsistent
                this->grad += 1.0 * out.grad; 
                other->grad += 1.0 * out.grad; 
            };

            this->_backward = _backward; 

            return out; 
        }    
};


int main() {
    // std::cout << "Hello, world!" << std::endl;
    Value a = Value(3.0); 
    // std::cout << "address of a: " << &a << std::endl;
    Value b = Value(4.0);
    // std::cout << "address of b: " << &b << std::endl;
    Value c = a.add(&b); 

    Value d = c.add(&b);

    std::cout << "d: " << d.data << std::endl;

    return 0;
}


/*

b: 4
this: 0x16d8d2c40
address of other: 0x16d8d2b88
address of a: 0x16d8d2c40
address of b: 0x16d8d2bf0
c: 7
*/

//if the address of b and other are not the same, my theory is that the set of parents will not point to the correct thing. How do i fix? 

