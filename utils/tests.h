#ifndef TESTS_H_
#define TESTS_H_

#include "neural_network.h"
#include "types.h"

int checkErrors(const arma::Mat<real>& Seq, const arma::Mat<real>& Par,
                std::ofstream& ofs, std::vector<real>& errors);

int checkNNErrors(NeuralNetwork& seq_nn, NeuralNetwork& par_nn,
                  std::string filename);

void BenchmarkGEMM();

#endif
