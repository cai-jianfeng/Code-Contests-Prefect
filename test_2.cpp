#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);

    // Read options
    int n = opt<int>("n");
    int t = opt<int>("t", 1);
    string positions = opt<string>("positions", "random");
    string sign = opt<string>("sign", "both");
    string ktype = opt<string>("ktype", "random");
    int k = opt<int>("k", -1);

    // Check the constraints
    if (n < 1 || n > 200000) {
        fprintf(stderr, "n must be between 1 and 200000\n");
        exit(1);
    }
    if (t < 1 || t > 10500) {
        fprintf(stderr, "t must be between 1 and 10500\n");
        exit(1);
    }

    int total_n = n * t;
    if (total_n > 200000) {
        fprintf(stderr, "Total n across all test cases should not exceed 200000\n");
        exit(1);
    }

    // Output t
    printf("%d\n", t);
    for (int test = 0; test < t; ++test) {
        int current_n = n;

        // Generate k
        int current_k;
        if (k != -1) {
            current_k = k;
        } else if (ktype == "random") {
            current_k = rnd.next(1, current_n);
        } else if (ktype == "1") {
            current_k = 1;
        } else if (ktype == "n") {
            current_k = current_n;
        } else {
            current_k = rnd.next(1, current_n);
        }

        // Make sure k is within [1, n]
        current_k = max(1, min(current_k, current_n));

        // Generate positions
        vector<int> x(current_n);

        if (positions == "random") {
            for (int i = 0; i < current_n; ++i) {
                x[i] = rnd.next(-1000000000, 1000000000);
            }
        } else if (positions == "same") {
            int value;
            if (sign == "positive") {
                value = rnd.next(1, 1000000000);
            } else if (sign == "negative") {
                value = rnd.next(-1000000000, -1);
            } else if (sign == "zero") {
                value = 0;
            } else {
                value = rnd.next(-1000000000, 1000000000);
            }
            for (int i = 0; i < current_n; ++i) {
                x[i] = value;
            }
        } else if (positions == "duplicate") {
            int num_unique = rnd.next(1, current_n);
            vector<int> unique_positions(num_unique);
            for (int i = 0; i < num_unique; ++i) {
                if (sign == "positive") {
                    unique_positions[i] = rnd.next(1, 1000000000);
                } else if (sign == "negative") {
                    unique_positions[i] = rnd.next(-1000000000, -1);
                } else if (sign == "zero") {
                    unique_positions[i] = 0;
                } else {
                    unique_positions[i] = rnd.next(-1000000000, 1000000000);
                }
            }
            for (int i = 0; i < current_n; ++i) {
                x[i] = unique_positions[rnd.next(0, num_unique - 1)];
            }
        } else if (positions == "small") {
            for (int i = 0; i < current_n; ++i) {
                x[i] = rnd.next(-10, 10);
            }
        } else if (positions == "big") {
            for (int i = 0; i < current_n; ++i) {
                if (sign == "positive") {
                    x[i] = rnd.next(1000000000 - 10, 1000000000);
                } else if (sign == "negative") {
                    x[i] = rnd.next(-1000000000, -1000000000 + 10);
                } else {
                    if (rnd.next(0, 1)) {
                        x[i] = rnd.next(1000000000 - 10, 1000000000);
                    } else {
                        x[i] = rnd.next(-1000000000, -1000000000 + 10);
                    }
                }
            }
        } else if (positions == "zero") {
            for (int i = 0; i < current_n; ++i) {
                x[i] = 0;
            }
        } else {
            for (int i = 0; i < current_n; ++i) {
                x[i] = rnd.next(-1000000000, 1000000000);
            }
        }

        // Apply sign constraints
        if (positions == "random" || positions == "small" || positions == "big" || positions == "duplicate") {
            if (sign == "positive") {
                for (int i = 0; i < current_n; ++i) {
                    x[i] = abs(x[i]);
                    if (x[i] == 0) x[i] = rnd.next(1, 1000000000);
                }
            } else if (sign == "negative") {
                for (int i = 0; i < current_n; ++i) {
                    x[i] = -abs(x[i]);
                    if (x[i] == 0) x[i] = rnd.next(-1000000000, -1);
                }
            } else if (sign == "zero") {
                for (int i = 0; i < current_n; ++i) {
                    x[i] = 0;
                }
            }
        }

        // Output the test case
        printf("%d %d\n", current_n, current_k);
        for (int i = 0; i < current_n; ++i) {
            printf("%d%c", x[i], i == current_n - 1 ? '\n' : ' ');
        }
    }

    return 0;
}
