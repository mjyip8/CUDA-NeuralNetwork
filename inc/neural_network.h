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

class MPIBatchInfo {
  public:
    int num_procs;
    int batch_size;
    int N;

    // whole batch info
    int batch;
    int last_col;
    int first_col;
    int total_batch_size;
    int total_batch_elems_X;
    int total_batch_elems_y;

    // minibatch info
    int minibatch_size;
    int last_batch_size;
    int minibatch_elems_X;
    int minibatch_elems_y;
    int* send_counts_X;
    int* send_counts_y;
    int* displacements_X;
    int* displacements_y;


    MPI_BatchInfo(int _num_procs, int _batch_size, _int batch, int _N) : 
            num_procs(_num_procs), batch_size(_batch_size), batch(_batch), N(_N) {
      send_counts_X = std::malloc(num_procs * sizeof(int));
      send_counts_y = std::malloc(num_procs * sizeof(int));
      displacements_X = std::calloc(num_procs, sizeof(int));
      displacements_y = std::calloc(num_procs, sizeof(int));
    }

    batchUpdate(const arma::Mat<real>& X, const arma::Mat<real>& y, const int _batch) {
      batch = _batch;
      first_col = batch_size * batch;
      last_col = std::min((batch + 1) * batch_size - 1, N - 1);

      total_batch_size = last_col - first_col + 1;
      total_batch_elems_X = total_batch_size * X.n_rows;
      total_batch_elems_y = total_batch_size * y.n_rows;

      minibatch_size = std::ceil((float) total_batch_size / (float) num_procs);
      last_batch_size = (total_batch_size % num_procs)? total_batch_size % num_procs : minibatch_size;
      minibatch_elems_X = minibatch_size * X.n_rows;
      minibatch_elems_y = minibatch_size * y.n_rows;

      std::memset(send_counts_X, minibatch_elems_X, (n_procs - 1) * sizeof(int));
      send_counts_X[num_procs - 1] = last_batch_size * X.n_rows;

      std::memset(send_counts_y, minibatch_elems_y, (n_procs - 1) * sizeof(int));
      send_counts_y[num_procs - 1] = last_batch_size * y.n_rows;

      # pragma unroll
      for (int i = 1; i < num_procs; i++) {
        displacements_X[i] = i * minibatch_elems_X;
        displacements_y[i] = i * minibatch_elems_y;
      }
    }

    ~MPI_BatchInfo(arma::Mat<real> X_batch, arma::Mat<real> y_batch) {
      std::free(send_counts_X);
      std::free(send_counts_y);
      std::free(displacements_X);
      std::free(displacements_y);
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

void initialize_grads(struct grads& grad, NeuralNetwork& nn);

#endif
