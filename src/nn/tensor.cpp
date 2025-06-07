#include "../../include/nn/tensor.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

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

const float &Tensor::operator()(std::size_t row, std::size_t col) const {
    if(_shape.size() != 2) {
        throw std::invalid_argument("This is not a 2D tensor. Use one index for 1D tensors.");
    }
    if(row >= _shape[0] || col >= _shape[1]) {
        throw std::invalid_argument("Index (" + std::to_string(row) + ", " + std::to_string(col) + ") is out of bounds for tensor of shape (" + std::to_string(_shape[0]) + ", " + std::to_string(_shape[1]) + ")");
    }
    return _data[row * _stride[0] + col * _stride[1]];
}

float &Tensor::operator()(std::size_t row, std::size_t col) {
    if(_shape.size() != 2) {
        throw std::invalid_argument("This is not a 2D tensor. Use one index for 1D tensors.");
    }
    if(row >= _shape[0] || col >= _shape[1]) {
        throw std::invalid_argument("Index (" + std::to_string(row) + ", " + std::to_string(col) + ") is out of bounds for tensor of shape (" + std::to_string(_shape[0]) + ", " + std::to_string(_shape[1]) + ")");
    }
    return _data[row * _stride[0] + col * _stride[1]];
}

const std::vector<std::size_t> &Tensor::shape() const {
    return this->_shape;
}

const std::vector<std::size_t> &Tensor::stride() const {
    return this->_stride;
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj){
    std::string string_repr = "[";
    if (obj.shape().size() == 0){
        os << obj.item();
        return os;
    } else if (obj.shape().size() == 1) {
        for (std::size_t i = 0; i < obj.shape()[0]; i++) {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1) {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    } else {
        for (std::size_t i = 0; i < obj.shape()[0]; i++) {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++) {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1) {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1) {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {

    // scalar + scalar
    if (this->_shape.size() == 0 && other->shape().size() == 0) {
        float result = this->item() + other->item();

        return std::make_shared<Tensor>(result);
    }

    // scalar + 1D tensor
    if(this->_shape.size() == 0 && other->shape().size() == 1) {
        std::vector<float> result(other->shape()[0]);
        for(std::size_t i = 0; i < other->shape()[0]; i++) {
            result[i] = (this->item() + ((*other)(i)));
        }
        return std::make_shared<Tensor>(result);
    }

    // scalar + 2D tensor
    if(this->_shape.size() == 0 && other->shape().size() == 2) {
        std::vector<std::vector<float>> result(other->shape()[0], std::vector<float>(other->shape()[1]));
        for(std::size_t i = 0; i < other->shape()[0]; i++) {
            for(std::size_t j = 0; j < other->shape()[1]; j++) {
                result[i][j] = this->item() + (*other)(i, j);
            }
        }
        return std::make_shared<Tensor>(result);    
    }

    // 1D tensor + scalar
    if(this->_shape.size() == 1 && other->shape().size() == 0) {
        std::vector<float> result(this->_shape[0]);
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            result[i] = ((*this)(i) + other->item());
        }
        return std::make_shared<Tensor>(result);
    }
    // 2D tensor + scalar
    if(this->_shape.size() == 2 && other->shape().size() == 0) {
        std::vector<std::vector<float>> result(this->_shape[0], std::vector<float>(this->_shape[1]));
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            for(std::size_t j = 0; j < this->_shape[1]; j++) {
                result[i][j] = (*this)(i, j) + other->item();
            }
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D tensor + 1D tensor
    if(this->_shape.size() == 1 && other->shape().size() == 1) {
        if(this->_shape[0] != other->shape()[0]) {
            throw std::invalid_argument("Shapes do not match for addition: " + std::to_string(this->_shape[0]) + " vs " + std::to_string(other->shape()[0]));
        }
        std::vector<float> result(this->_shape[0]);
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            result[i] = (*this)(i) + (*other)(i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D tensor + 2D tensor
    if(this->_shape.size() == 2 && other->shape().size() == 2) {
        if(this->_shape[0] != other->shape()[0] || this->_shape[1] != other->shape()[1]) {
            throw std::invalid_argument("Shapes do not match for addition: (" + std::to_string(this->_shape[0]) + ", " + std::to_string(this->_shape[1]) + ") vs (" + std::to_string(other->shape()[0]) + ", " + std::to_string(other->shape()[1]) + ")");
        }
        std::vector<std::vector<float>> result(this->_shape[0], std::vector<float>(this->_shape[1]));
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            for(std::size_t j = 0; j < this->_shape[1]; j++) {
                result[i][j] = (*this)(i, j) + (*other)(i, j);
            }
        }
        return std::make_shared<Tensor>(result);
    }
    
    throw std::invalid_argument("Unsupported tensor shapes for addition: (" + std::to_string(this->_shape.size()) + ") vs (" + std::to_string(other->shape().size()) + ")");

}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {

    if(this->_shape.size() == 0 && other->shape().size() == 0) {
        float result = this->item() * other->item();
        return std::make_shared<Tensor>(result);
    }

    if(this->_shape[this->_shape.size() - 1] != other->shape()[0]) {
        throw std::invalid_argument("Shapes do not match for multiplication: (" + std::to_string(this->_shape[0]) + ", " + std::to_string(this->_shape[1]) + ") vs (" + std::to_string(other->shape()[0]) + ", " + std::to_string(other->shape()[1]) + ")");
    }

    // 1D tensor * 1D tensor
    if(this->_shape.size() == 1 && other->shape().size() == 1) {
        float result = 0;
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            result += (*this)(i) * (*other)(i);
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D tensor * 1D tensor
    if(this->_shape.size() == 2 && other->shape().size() == 1) {
        std::vector<float> result(this->_shape[0]);
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            float sum = 0;
            for(std::size_t j = 0; j < this->_shape[1]; j++) {
                sum += (*this)(i, j) * (*other)(j);
            }
            result[i] = sum;
        }
        return std::make_shared<Tensor>(result);
    }

    // 1D tensor * 2D tensor
    if(this->_shape.size() == 1 && other->shape().size() == 2) {
        std::vector<float> result(other->shape()[1]);
        for(std::size_t j = 0; j < other->shape()[1]; j++) {
            float sum = 0;
            for(std::size_t i = 0; i < this->_shape[0]; i++) {
                sum += (*this)(i) * (*other)(i, j);
            }
            result[j] = sum;
        }
        return std::make_shared<Tensor>(result);
    }

    // 2D tensor * 2D tensor
    if(this->_shape.size() == 2 && other->shape().size() == 2) {
    std::vector<std::vector<float>> result(this->_shape[0], std::vector<float>(other->shape()[1], 0.0f));
        for(std::size_t i = 0; i < this->_shape[0]; i++) {
            std::vector<float> result_row(other->shape()[1], 0.);
            for(std::size_t j = 0; j < other->shape()[1]; j++) {
                float sum = 0;
                for(std::size_t k = 0; k < this->_shape[1]; k++) {
                    sum += (*this)(i, k) * (*other)(k, j);
                }
                result_row[j] = sum;
            }
            result[i] = result_row;
        }
        return std::make_shared<Tensor>(result);
    }
}