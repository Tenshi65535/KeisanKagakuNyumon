#include <iostream>
#include <iomanip>
#include <vector>
#include <lapacke.h>

void printSolution(const std::vector<double>& solution) {
    int n = solution.size();
    std::cout << std::fixed;
    std::cout << std::setprecision(16);
    std::cout << "Solution vector:\n";
    for (int i = 0; i < n; i++)
    {
        std::cout << "x[" << i << "] = " << solution[i] << std::endl;
    }
}

int main() {
    int n;
    std::cout << "Enter the size of the Hilbert matrix: ";
    std::cin >> n;

    std::vector<double> b(n, 1.0);

    std::vector<double> hilbertMatrix(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            hilbertMatrix[i * n + j] = 1.0 / (i + j + 1);
        }
    }

    std::vector<double> solution(n);
    int info;
    lapack_int* ipiv = new lapack_int[n];
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, hilbertMatrix.data(), n, ipiv, b.data(), 1);

    if (info == 0) {
        for (int i = 0; i < n; ++i) {
            solution[i] = b[i];
        }
        printSolution(solution);
    } else {
        std::cout << "Failed to solve the equation." << std::endl;
    }

    delete[] ipiv;
    return 0;
}
