#ifndef UTILS_COMMON_H
#define UTILS_COMMON_H

#include <armadillo>
#include <cassert>
#include <string>

#include "types.h"

#define ASSERT_MAT_SAME_SIZE(mat1, mat12) \
  assert(mat1.n_rows == mat2.n_rows && mat1.n_cols == mat2.n_cols)

struct grads {
  std::vector<arma::Mat<real>> dW;
  std::vector<arma::Col<real>> db;
};

struct cache {
  arma::Mat<real> X;
  std::vector<arma::Mat<real>> z;
  std::vector<arma::Mat<real>> a;
  arma::Mat<real> yc;
};

/*
 * Applies the sigmoid function to each element of the matrix
 * and returns a new matrix.
 */
void sigmoid(const arma::Mat<real>& mat, arma::Mat<real>& mat2);

/*
 * ReLU activation
 */
void relu(const arma::Mat<real>& mat, arma::Mat<real>& mat2);

/*
 * Applies the softmax to each rowvec of the matrix
 */
void softmax(const arma::Mat<real>& mat, arma::Mat<real>& mat2);

/*
 * Performs gradient check by comparing numerical and analytical gradients.
 */
bool gradcheck(struct grads& grads1, struct grads& grads2);

/*
 * Compares the two label vectors to compute precision.
 */
real precision(arma::Row<real> vec1, arma::Row<real> vec2);

/*
 * Converts label vector into a matrix of one-hot label vectors
 * @params label : label vector
 * @params C : Number of classes
 * @params [out] y : The y matrix.
 */
void label_to_y(arma::Row<real> label, int C, arma::Mat<real>& y);

void save_label(std::string filename, arma::Row<real>& label);

#endif
