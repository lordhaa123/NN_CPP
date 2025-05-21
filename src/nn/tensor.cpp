#include "../../include/nn/tensor.h"
#include <iostream>
#include <vector>
#include <string>

Tensor::Tensor(float data) : _data{data}, _shape{} {};

Tensor::Tensor(std::vector<float> data) : _data{data}, _shape{data.size()}, _stride{1} {};

Tensor::Tensor(std::vector<std::vector<float>> data) : _shape{data.size(), data[0].size()}, _stride{data[0].size(), 1} {
    std::size_t expected_col_size = data[0].size();
    for(const auto &row : data) {
        if(row.size() != expected_col_size) {
            throw std::invalid_argument("Dimentions of the 2D vector are not consistent");
        }
    }

    for(const auto &row : data) {
        _data.insert(_data.end(), row.begin(), row.end());
    }
}

const float &Tensor::item() const {
    if(_data.size() == 1) {
        return _data[0];
    } else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

float &Tensor::item() {
    if(_data.size() == 1) {
        return _data[0];
    } else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const float &Tensor::operator()(std::size_t i) const {
    if(_shape.size() == 0) {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if(_shape.size() == 1) {
        if(i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

float &Tensor::operator()(std::size_t i) {
    if(_shape.size() == 0) {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if(_shape.size() == 1) {
        if(i >= _shape[0]) {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}



