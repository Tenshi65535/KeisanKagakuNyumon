#include <iostream>
#include <gmp.h>
#include <gmpxx.h>

void generateHilbertMatrix(int n) {
    mpf_set_default_prec(1024);  // Set the precision of the GMP library

    mpf_t** hilbertMatrix = new mpf_t*[n];
    for (int i = 0; i < n; ++i) {
        hilbertMatrix[i] = new mpf_t[n];
        for (int j = 0; j < n; ++j) {
            mpf_init2(hilbertMatrix[i][j], 1024);
            mpf_set_ui(hilbertMatrix[i][j], 1);

            mpf_t temp;
            mpf_init2(temp, 1024);
            mpf_set_ui(temp, i + j + 1);

            mpf_div(hilbertMatrix[i][j], hilbertMatrix[i][j], temp);
            mpf_clear(temp);
        }
    }

    // Create the all-one vector
    mpf_t* onesVector = new mpf_t[n];
    for (int i = 0; i < n; ++i) {
        mpf_init2(onesVector[i], 1024);
        mpf_set_ui(onesVector[i], 1);
    }

    // Solve the system of equations using Gauss-Jordan elimination
    for (int k = 0; k < n; ++k) {
        // Find the row with the maximum value for column k
        int maxRow = k;
        mpf_t maxValue;
        mpf_init2(maxValue, 1024);
        mpf_abs(maxValue, hilbertMatrix[k][k]);

        for (int i = k + 1; i < n; ++i) {
            mpf_t temp;
            mpf_init2(temp, 1024);
            mpf_abs(temp, hilbertMatrix[i][k]);

            if (mpf_cmp(temp, maxValue) > 0) {
                mpf_set(maxValue, temp);
                maxRow = i;
            }

            mpf_clear(temp);
        }

        // Swap the rows if necessary
        if (maxRow != k) {
            for (int j = k; j < n; ++j) {
                mpf_swap(hilbertMatrix[k][j], hilbertMatrix[maxRow][j]);
            }
            mpf_swap(onesVector[k], onesVector[maxRow]);
        }

        // Perform row operations to eliminate the elements below and above the pivot
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                mpf_t ratio;
                mpf_t temp;
                mpf_init2(ratio, 1024);
                mpf_init2(temp, 1024);
                mpf_div(ratio, hilbertMatrix[i][k], hilbertMatrix[k][k]);

                for (int j = k; j < n; ++j) {
                    mpf_t temp;
                    mpf_init2(temp, 1024);
                    mpf_mul(temp, ratio, hilbertMatrix[k][j]);
                    mpf_sub(hilbertMatrix[i][j], hilbertMatrix[i][j], temp);
                    mpf_clear(temp);
                }

                mpf_mul(temp, ratio, onesVector[k]);
                mpf_sub(onesVector[i], onesVector[i], temp);

                mpf_clear(ratio);
            }
        }
    }

    // Normalize the solution vector
    for (int i = 0; i < n; ++i) {
        mpf_div(onesVector[i], onesVector[i], hilbertMatrix[i][i]);
    }

    // Print the solution vector
    for (int i = 0; i < n; ++i) {
        gmp_printf("%.16Ff\n", onesVector[i]);
    }

    // Cleanup
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mpf_clear(hilbertMatrix[i][j]);
        }
        delete[] hilbertMatrix[i];
        mpf_clear(onesVector[i]);
    }
    delete[] hilbertMatrix;
    delete[] onesVector;
}

int main() {
    int n;
    std::cout << "Enter the size of the Hilbert matrix: ";
    std::cin >> n;
    generateHilbertMatrix(n);
    return 0;
}
