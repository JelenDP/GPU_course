#include <vector>
#include <future>

template<typename T>
auto cpu_scalar_prod_naive(std::vector<T> const& A, std::vector<T> const& B, int N) 
{  
    auto c = 0.0;

    for(int i=0; i<N; ++i)
    {
        c += A[i] * B[i];
    }

    return c;
}

template<typename T>
auto cpu_scalar_prod_parallel(std::vector<T> const& A, std::vector<T> const& B, int N) 
{
    // cpu parallel implementation
    int n = std::thread::hardware_concurrency();
    std::vector<std::future<double>> futures(n);
    
    auto cpu_scalar_prod_elementary = [](std::vector<T> const& A, std::vector<T> const& B, int start, int end) 
    {
        double sum = 0.0;
        for ( int i = start; i < end; ++i)
        {
            sum += A[i] * B[i]; 
        }
        return sum;
    };
    for ( int k=0; k<n; ++k ) 
    {
        int size = static_cast<int>(A.size());
        int start = k     * size / n;
        int end   = (k+1) * size / n;
        futures[k] = std::async(std::launch::async, cpu_scalar_prod_elementary, std::cref(A), std::cref(B), start, end);
    }
    
    auto result = std::accumulate(
                        futures.begin(),futures.end(),0.0,
                        [](double acc, std::future<double>& f){return acc+ f.get();}
                        );
    return result;
}

