#include <iostream>
#include <mpfr.h>

void generateHilbertMatrix(int n) {
    mpfr_set_default_prec(1024);  // Set the precision of the mpfr library

    mpfr_t** hilbertMatrix = new mpfr_t*[n];
    for (int i = 0; i < n; ++i) {
        hilbertMatrix[i] = new mpfr_t[n];
        for (int j = 0; j < n; ++j) {
            mpfr_init2(hilbertMatrix[i][j], 1024);
            mpfr_set_ui(hilbertMatrix[i][j], 1, MPFR_RNDN);

            mpfr_t temp;
            mpfr_init2(temp, 1024);
            mpfr_set_ui(temp, i + j + 1, MPFR_RNDN);

            mpfr_div(hilbertMatrix[i][j], hilbertMatrix[i][j], temp, MPFR_RNDN);
            mpfr_clear(temp);
        }
    }

    // Create the all-one vector
    mpfr_t* onesVector = new mpfr_t[n];
    for (int i = 0; i < n; ++i) {
        mpfr_init2(onesVector[i], 1024);
        mpfr_set_ui(onesVector[i], 1, MPFR_RNDN);
    }

    // Solve the system of equations using Gauss-Jordan elimination
    for (int k = 0; k < n; ++k) {
        // Find the row with the maximum value for column k
        int maxRow = k;
        mpfr_t maxValue;
        mpfr_init2(maxValue, 1024);
        mpfr_abs(maxValue, hilbertMatrix[k][k], MPFR_RNDN);

        for (int i = k + 1; i < n; ++i) {
            mpfr_t temp;
            mpfr_init2(temp, 1024);
            mpfr_abs(temp, hilbertMatrix[i][k], MPFR_RNDN);

            if (mpfr_cmp(temp, maxValue) > 0) {
                mpfr_set(maxValue, temp, MPFR_RNDN);
                maxRow = i;
            }

            mpfr_clear(temp);
        }

        // Swap the rows if necessary
        if (maxRow != k) {
            for (int j = k; j < n; ++j) {
                mpfr_swap(hilbertMatrix[k][j], hilbertMatrix[maxRow][j]);
            }
            mpfr_swap(onesVector[k], onesVector[maxRow]);
        }

        // Perform row operations to eliminate the elements below and above the pivot
        for (int i = 0; i < n; ++i) {
            if (i != k) {
                mpfr_t ratio;
                mpfr_t temp;
                mpfr_init2(ratio, 1024);
                mpfr_init2(temp, 1024);
                mpfr_div(ratio, hilbertMatrix[i][k], hilbertMatrix[k][k], MPFR_RNDN);

                for (int j = k; j < n; ++j) {
                    mpfr_t temp;
                    mpfr_init2(temp, 1024);
                    mpfr_mul(temp, ratio, hilbertMatrix[k][j], MPFR_RNDN);
                    mpfr_sub(hilbertMatrix[i][j], hilbertMatrix[i][j], temp, MPFR_RNDN);
                    mpfr_clear(temp);
                }

                mpfr_mul(temp, ratio, onesVector[k], MPFR_RNDN);
                mpfr_sub(onesVector[i], onesVector[i], temp, MPFR_RNDN);

                mpfr_clear(ratio);
            }
        }
    }

    // Normalize the solution vector
    for (int i = 0; i < n; ++i) {
        mpfr_div(onesVector[i], onesVector[i], hilbertMatrix[i][i], MPFR_RNDN);
    }

    // Print the solution vector
    for (int i = 0; i < n; ++i) {
        mpfr_printf("%.16Rf\n", onesVector[i]);
    }

    // Cleanup
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mpfr_clear(hilbertMatrix[i][j]);
        }
        delete[] hilbertMatrix[i];
        mpfr_clear(onesVector[i]);
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