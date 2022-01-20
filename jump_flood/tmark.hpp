#include <chrono>

auto tmark()
{
    return std::chrono::high_resolution_clock::now();
}
template<typename T1, typename T2>

auto delta_time( T1&& t1, T2&& t2)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
}