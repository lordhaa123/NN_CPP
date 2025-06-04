#include "../src/nn/tensor.cpp"
#include <iostream>

int main() {
    bool all_tests_passed = true;
    std::cout << "Running Tensor tests..." << std::endl;
    // Test with a single float
    {
        Tensor singleTensor(3.14f);
        if(singleTensor.item() == 3.14f) {
            std::cout << "PASS : " << "Single Tensor item: " << singleTensor.item() << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 3.14f but got " << singleTensor.item() << std::endl;
        }
        singleTensor.item() = 2.71f; // Modify the item
        if(singleTensor.item() == 2.71f) {
            std::cout << "PASS : " << "Modified Single Tensor item: " << singleTensor.item() << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 2.71f but got " << singleTensor.item() << std::endl;
        }
        // check shape and stride
        if(singleTensor.shape().size() == 0) {
            std::cout << "PASS : " << "Single Tensor shape is empty as expected." << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected empty shape for single tensor but got size " << singleTensor.shape().size() << std::endl;
        }
        if(singleTensor.stride().size() == 0) {
            std::cout << "PASS : " << "Single Tensor stride is empty as expected." << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected empty stride for single tensor but got size " << singleTensor.stride().size() << std::endl;
        }
    }

    // Test with a vector of floats
    {
        Tensor vectorTensor({1.2f, 2.3f, 3.4f});
        if(vectorTensor(0) == 1.2f && vectorTensor(1) == 2.3f && vectorTensor(2) == 3.4f) {
            std::cout<< "PASS : " << "Vector Tensor items: " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
        }
        else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 1.2f, 2.3f, 3.4f but got " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
        }
        vectorTensor(0) = 4.5f; // Modify the first item
        vectorTensor(1) = 5.3f; // Modify the second item
        if(vectorTensor(0) == 4.5f && vectorTensor(1) == 5.3f && vectorTensor(2) == 3.4f) {
            std::cout << "PASS : " << "Modified Vector Tensor items: " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 4.5f, 5.3f, 3.4f but got " << vectorTensor(0) << ", " << vectorTensor(1) << ", " << vectorTensor(2) << std::endl;
        }
        // check shape and stride
        if(vectorTensor.shape().size() == 1 && vectorTensor.shape()[0] == 3) {
            std::cout << "PASS : " << "Vector Tensor shape is correct: " << vectorTensor.shape()[0] << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected shape [3] but got " << vectorTensor.shape().size() << std::endl;
        }
        if(vectorTensor.stride().size() == 1 && vectorTensor.stride()[0] == 1) {
            std::cout << "PASS : " << "Vector Tensor stride is correct: " << vectorTensor.stride()[0] << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected stride [1] but got " << vectorTensor.stride().size() << std::endl;
        }
    }

    // Test with a 2D vector
    {
        Tensor matrixTensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
        if(matrixTensor(0, 0) == 1.0f && matrixTensor(0, 1) == 2.0f &&
        matrixTensor(1, 0) == 3.0f && matrixTensor(1, 1) == 4.0f) {
            std::cout << "PASS : " << "Matrix Tensor items: " << matrixTensor(0, 0) << ", " << matrixTensor(0, 1) << ", "
                    << matrixTensor(1, 0) << ", " << matrixTensor(1, 1) << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 1.0f, 2.0f, 3.0f, 4.0f but got "
                    << matrixTensor(0, 0) << ", " << matrixTensor(0, 1) << ", "
                    << matrixTensor(1, 0) << ", " << matrixTensor(1, 1) << std::endl;
        }
        matrixTensor(0, 0) = 5.0f; // Modify the first item
        matrixTensor(1, 1) = 6.0f; // Modify the last item
        if(matrixTensor(0, 0) == 5.0f && matrixTensor(0, 1) == 2.0f &&
        matrixTensor(1, 0) == 3.0f && matrixTensor(1, 1) == 6.0f) {
            std::cout << "PASS : " << "Modified Matrix Tensor items: " << matrixTensor(0, 0) << ", "
                    << matrixTensor(0, 1) << ", " << matrixTensor(1, 0) << ", " << matrixTensor(1, 1) << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 5.0f, 2.0f, 3.0f, 6.0f but got "
                    << matrixTensor(0, 0) << ", " << matrixTensor(0, 1) << ", "
                    << matrixTensor(1, 0) << ", " << matrixTensor(1, 1) << std::endl;
        }
        // check shape and stride
        if(matrixTensor.shape().size() == 2 && matrixTensor.shape()[0] == 2 && matrixTensor.shape()[1] == 2) {
            std::cout << "PASS : " << "Matrix Tensor shape is correct: " << matrixTensor.shape()[0] << ", " << matrixTensor.shape()[1] << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected shape [2, 2] but got " << matrixTensor.shape().size() << std::endl;
        }
        if(matrixTensor.stride().size() == 2 && matrixTensor.stride()[0] == 2 && matrixTensor.stride()[1] == 1) {
            std::cout << "PASS : " << "Matrix Tensor stride is correct: " << matrixTensor.stride()[0] << ", " << matrixTensor.stride()[1] << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected stride [2, 1] but got " << matrixTensor.stride().size() << std::endl;
        }
    }

    // std::cout<< "Testing output operator for single tensor: " << singleTensor << std::endl;
    // std::cout<< "Testing output operator for vector tensor: " << vectorTensor << std::endl;
    // std::cout<< "Testing output operator for matrix tensor: " << matrixTensor << std::endl;

    // tests for tensor addition
    {
        // testing scalar + scalar
        std::shared_ptr<Tensor> scalar1 = std::make_shared<Tensor>(2.0f);
        std::shared_ptr<Tensor> scalar2 = std::make_shared<Tensor>(3.0f);

        std::shared_ptr<Tensor> scalar_sum = *scalar1 + scalar2;

        if(scalar_sum->item() == 5.0f) {
            std::cout << "PASS : " << "Scalar addition result: " << scalar_sum->item() << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected 5.0f but got " << scalar_sum->item() << std::endl;
        }

        // testing scalar + 1D tensor
        std::shared_ptr<Tensor> scalar3 = std::make_shared<Tensor>(1.0f);
        std::shared_ptr<Tensor> vector1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
        std::shared_ptr<Tensor> vector_sum = *scalar3 + vector1;
        if(vector_sum->shape().size() == 1 && vector_sum->shape()[0] == 3 && (*vector_sum)(0) == 2.0f && (*vector_sum)(1) == 3.0f &&
           (*vector_sum)(2) == 4.0f) {
            std::cout << "PASS : " << "Scalar + Vector addition result: " << *vector_sum << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [2.0, 3.0, 4.0] but got " << *vector_sum << std::endl;
        }

        // testing scalar + 2D tensor
        std::shared_ptr<Tensor> scalar4 = std::make_shared<Tensor>(2.0f);
        std::shared_ptr<Tensor> matrix1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}, {3.0f, 4.0f, 5.0f}});
        std::shared_ptr<Tensor> matrix_sum = *scalar4 + matrix1;
        if(matrix_sum->shape().size() == 2 && matrix_sum->shape()[0] == 2 && matrix_sum->shape()[1] == 3 &&
           (*matrix_sum)(0, 0) == 3.0f && (*matrix_sum)(0, 1) == 4.0f && (*matrix_sum)(0, 2) == 5.0f &&
           (*matrix_sum)(1, 0) == 5.0f && (*matrix_sum)(1, 1) == 6.0f && (*matrix_sum)(1, 2) == 7.0f) {
            std::cout << "PASS : " << "Scalar + Matrix addition result: " << *matrix_sum << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]] but got " << *matrix_sum << std::endl;
        }

        // testing 1D tensor + scalar
        std::shared_ptr<Tensor> vector2 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
        std::shared_ptr<Tensor> scalar5 = std::make_shared<Tensor>(2.0f);
        std::shared_ptr<Tensor> vector_sum2 = *vector2 + scalar5;
        if(vector_sum2->shape().size() == 1 && vector_sum2->shape()[0] == 3 && (*vector_sum2)(0) == 3.0f && (*vector_sum2)(1) == 4.0f &&
           (*vector_sum2)(2) == 5.0f) {
            std::cout << "PASS : " << "Vector + Scalar addition result: " << *vector_sum2 << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [3.0, 4.0, 5.0] but got " << *vector_sum2 << std::endl;
        }

        // testing 2D tensor + scalar
        std::shared_ptr<Tensor> matrix2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f, 3.0f}, {3.0f, 4.0f, 5.0f}});
        std::shared_ptr<Tensor> scalar6 = std::make_shared<Tensor>(2.0f);
        std::shared_ptr<Tensor> matrix_sum2 = *matrix2 + scalar6;
        if(matrix_sum2->shape().size() == 2 && matrix_sum2->shape()[0] == 2 && matrix_sum2->shape()[1] == 3 &&
           (*matrix_sum2)(0, 0) == 3.0f && (*matrix_sum2)(0, 1) == 4.0f && (*matrix_sum2)(0, 2) == 5.0f &&
           (*matrix_sum2)(1, 0) == 5.0f && (*matrix_sum2)(1, 1) == 6.0f && (*matrix_sum2)(1, 2) == 7.0f) {
            std::cout << "PASS : " << "Matrix + Scalar addition result: " << *matrix_sum2 << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]] but got " << *matrix_sum2 << std::endl;
        }

        // testing 1D tensor + 1D tensor
        std::shared_ptr<Tensor> vector3 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
        std::shared_ptr<Tensor> vector4 = std::make_shared<Tensor>(std::vector<float>{4.0f, 5.0f, 6.0f});
        std::shared_ptr<Tensor> vector_sum3 = *vector3 + vector4;
        if(vector_sum3->shape().size() == 1 && vector_sum3->shape()[0] == 3 && (*vector_sum3)(0) == 5.0f && (*vector_sum3)(1) == 7.0f &&
           (*vector_sum3)(2) == 9.0f) {
            std::cout << "PASS : " << "Vector + Vector addition result: " << *vector_sum3 << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [5.0, 7.0, 9.0] but got " << *vector_sum3 << std::endl;
        }

        // testing 2D tensor + 2D tensor
        std::shared_ptr<Tensor> matrix3 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});
        std::shared_ptr<Tensor> matrix4 = std::make_shared<Tensor>(std::vector<std::vector<float>>{{5.0f, 6.0f}, {7.0f, 8.0f}});
        std::shared_ptr<Tensor> matrix_sum3 = *matrix3 + matrix4;
        if(matrix_sum3->shape().size() == 2 && matrix_sum3->shape()[0] == 2 && matrix_sum3->shape()[1] == 2 &&
           (*matrix_sum3)(0, 0) == 6.0f && (*matrix_sum3)(0, 1) == 8.0f &&
           (*matrix_sum3)(1, 0) == 10.0f && (*matrix_sum3)(1, 1) == 12.0f) {
            std::cout << "PASS : " << "Matrix + Matrix addition result: " << *matrix_sum3 << std::endl;
        } else {
            all_tests_passed = false;
            std::cerr << "FAIL : " << "Error: Expected [[6.0, 8.0], [10.0, 12.0]] but got " << *matrix_sum3 << std::endl;
        }

    }

    std::cout<<std::endl;
    if(all_tests_passed) {
        std::cout << "All Tensor tests passed!" << std::endl;
    } else {
        std::cerr << "Some Tensor tests failed." << std::endl;
    }

    return 0;
}