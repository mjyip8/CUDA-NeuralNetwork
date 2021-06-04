#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cmath>
#include <iostream>

#include "../utils/types.h"

class NeuralNetwork {
 public:
  const int num_layers = 2;
  // H[i] is the number of neurons in layer i (where i=0 implies input layer)
  std::vector<int> H;
  // Weights of the neural network
  // W[i] are the weights of the i^th layer
  std::vector<arma::Mat<real>> W;
  // Biases of the neural network
  // b[i] is the row vector biases of the i^th layer
  std::vector<arma::Col<real>> b;

  NeuralNetwork(std::vector<int> _H) {
    W.resize(num_layers);
    b.resize(num_layers);
    H = _H;

    for (int i = 0; i < num_layers; i++) {
      arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
      W[i] = 0.01 * arma::randn<arma::Mat<real>>(H[i + 1], H[i]);
      b[i].zeros(H[i + 1]);
    }
  }
};

void feedforward(NeuralNetwork& nn, const arma::Mat<real>& X,
                 struct cache& bpcache);
real loss(NeuralNetwork& nn, const arma::Mat<real>& yc,
          const arma::Mat<real>& y, real reg);
void backprop(NeuralNetwork& nn, const arma::Mat<real>& y, real reg,
              const struct cache& bpcache, struct grads& bpgrads);
void numgrad(NeuralNetwork& nn, const arma::Mat<real>& X,
             const arma::Mat<real>& y, real reg, struct grads& numgrads);
void train(NeuralNetwork& nn, const arma::Mat<real>& X,
           const arma::Mat<real>& y, real learning_rate, real reg = 0.0,
           const int epochs = 15, const int batch_size = 800,
           bool grad_check = false, int print_every = -1, int debug = 0);
void predict(NeuralNetwork& nn, const arma::Mat<real>& X,
             arma::Row<real>& label);

void parallel_train(NeuralNetwork& nn, const arma::Mat<real>& X,
                    const arma::Mat<real>& y, real learning_rate,
                    real reg = 0.0, const int epochs = 15,
                    const int batch_size = 800, bool grad_check = false,
                    int print_every = -1, int debug = 0);

#endif
