#include <vector>
#include <random>
#include <numeric>
#include <future>
#include <iostream>

#include "tmark.hpp"
#include "cpu_scalar_prod.hpp"

int main(){

    static const int N = 10'000'000;
    static const int bs = 64; //block size

    std::vector<double> A(N);
    std::vector<double> B(N); 

    std::mt19937 mersenne_engine{42};  // Generates random integers
    std::uniform_real_distribution<float> dist{-0.1f, 0.1f};
    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    generate(A.begin(), A.end(), gen);
    generate(B.begin(), B.end(), gen);

    //naive implementation
    auto t0 = tmark();
    auto prod = cpu_scalar_prod_naive(A, B, N);
    auto t1 = tmark();

    //cpu parallel implementation
    int n = std::thread::hardware_concurrency();
    std::vector<std::future<double>> futures(n);
    
    auto cpu_scalar_prod_parallel = [](std::vector<double> const& A, std::vector<double> const& B, int start, int end) 
    {
        double sum = 0.0;
        for ( int i = start; i < end; ++i)
        {
            sum += A[i] * B[i]; 
        }
        return sum;
    };

    auto t0_parallel = tmark();
    for ( int k=0; k<n; ++k ) 
    {
        int size = static_cast<int>(A.size());
        int start = k     * size / n;
        int end   = (k+1) * size / n;
        futures[k] = std::async(std::launch::async, cpu_scalar_prod_parallel, A, B, start, end);
    }
    
    auto prod_parallel = std::accumulate(
                        futures.begin(),futures.end(),0.0,
                        [](double acc, std::future<double>& f){return acc+ f.get();}
                        );
    auto t1_parallel = tmark();

    std::cout << "Results of naive:    " << prod          << std::endl; 
    std::cout << "Results of parallel: " << prod_parallel << std::endl; 
    std::cout << "Results of std:      " << std::inner_product(std::begin(A), std::end(A), std::begin(B), 0.0) << std::endl;
    std::cout << "CPU time of naive    " << delta_time(t0,t1) << " ms\n";
    std::cout << "CPU time of parallel " << delta_time(t0_parallel,t1_parallel) << " ms" << "; number of threads: " << n << std::endl;

    return 0;
}