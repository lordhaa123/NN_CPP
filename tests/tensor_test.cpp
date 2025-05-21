#include "../src/nn/tensor.cpp"
#include <iostream>

int main() {
    // Test with a single float
    Tensor singleTensor(3.14f);
    if(singleTensor.item() == 3.14f) {
        std::cout << "PASS : " << "Single Tensor item: " << singleTensor.item() << std::endl;
    } else {
        std::cerr << "FAIL : " << "Error: Expected 3.14f but got " << singleTensor.item() << std::endl;
    }
    singleTensor.item() = 2.71f; // Modify the item
    if(singleTensor.item() == 2.71f) {
        std::cout << "PASS : " << "Modified Single Tensor item: " << singleTensor.item() << std::endl;
    } else {
        std::cerr << "FAIL : " << "Error: Expected 2.71f but got " << singleTensor.item() << std::endl;
    }

    // Test with a vector of floats
    Tensor vectorTensor({1.2f, 2.3f, 3.4f});
    if(vectorTensor(0) == 1.2f && vectorTensor(1) == 2.3f && vectorTensor(2) == 3.4f) {
        std::cout<< "PASS : " << "Vector Tensor items: " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
    }
    else {
        std::cerr << "FAIL : " << "Error: Expected 1.2f, 2.3f, 3.4f but got " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
    }
    vectorTensor(0) = 4.5f; // Modify the first item
    vectorTensor(1) = 5.3f; // Modify the second item
    if(vectorTensor(0) == 4.5f && vectorTensor(1) == 5.3f && vectorTensor(2) == 3.4f) {
        std::cout << "PASS : " << "Modified Vector Tensor items: " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
    } else {
        std::cerr << "FAIL : " << "Error: Expected 4.5f, 5.3f, 3.4f but got " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
    }

    return 0;
}