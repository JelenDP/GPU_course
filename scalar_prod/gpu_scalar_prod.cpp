#include <CL/cl2.hpp>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <numeric>      // std::accumulate

//Own
#include "tmark.hpp"
#include "cpu_scalar_prod.hpp"

int main()
{
    try
    {
        // User defined input
        const std::size_t N = 20'000'000;
        std::vector<cl_float> a_vec(N), b_vec(N), c_vec(N, 0.0);

        // Fill vectors with random values between -0.1 and 0.1
        std::mt19937 mersenne_engine{42};  // Generates random integers
        std::uniform_real_distribution<float> dist{-0.1f, 0.1f};
        auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
        generate(a_vec.begin(), a_vec.end(), gen);
        generate(b_vec.begin(), b_vec.end(), gen);
        
        // Open-CL part 
        cl::CommandQueue queue = cl::CommandQueue::getDefault();

        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Platform platform{device.getInfo<CL_DEVICE_PLATFORM>()};

        std::cout << "Default queue on platform: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "Default queue on device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        auto kernel_op = "float op(float a, float b) { return a + b; }";
        auto host_op = [](float a, float b){ return a + b; };
        cl_float zero_elem = 0.0;
        
        // Load program source
        std::ifstream source_file{ "./../../scalar_prod/scalar_prod.cl" };
        if (!source_file.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "./../../scalar_prod.cl" };

        // Create program 
        cl::Program program{ std::string{ std::istreambuf_iterator<char>{ source_file },
                                          std::istreambuf_iterator<char>{} }.append(kernel_op) };
        program.build({ device });

        // Create kernels
        // First: multiplication by element
        auto scalar_prod = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(program, "scalar_prod");
        // Second: reduce the result vector to scalar with summation
        auto reduce = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl_uint, cl_float>(program, "reduce");

        // Max size of work group        
        auto wgs = reduce.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
       
        // Decrease size of work group as size of local memory
        while (device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() < wgs * 2 * sizeof(cl_float))
            wgs -= reduce.getKernel().getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

        if (wgs == 0) throw std::runtime_error{"Not enough local memory to serve a single sub-group."};

        auto factor = wgs * 2;
        // Every pass reduces input length by 'factor'.
        // If actual size is not divisible by factor,
        // an extra output element is produced using some
        // number of zero_elem inputs.
        auto new_size = [factor](const std::size_t actual)
        {
            return actual / factor + (actual % factor == 0 ? 0 : 1);
        };
        // NOTE: because one work-group produces one output
        //       new_size == number_of_work_groups
        auto global = [=](const std::size_t actual){ return new_size(actual) * wgs; };
        
        // Create buffers
        cl::Buffer a_buf{ context, std::begin(a_vec), std::end(a_vec), true },
                   b_buf{ context, std::begin(b_vec), std::end(b_vec), true },
                   c_buf{ context, std::begin(c_vec), std::end(c_vec), false },
                   red_buf{ context, CL_MEM_READ_WRITE, new_size(N) * sizeof(cl_float) };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(a_vec), std::end(a_vec), a_buf);
        cl::copy(queue, std::begin(b_vec), std::end(b_vec), b_buf);
        cl::copy(queue, std::begin(c_vec), std::end(c_vec), c_buf);

        // Launch kernels
        auto start_gpu = tmark();
        cl::Event scalar_prod_kernel{ scalar_prod(cl::EnqueueArgs{ queue, cl::NDRange{ N } }, a_buf, b_buf, c_buf) };
        scalar_prod_kernel.wait();
        //cl::copy(queue, c_buf, std::begin(c_vec), std::end(c_vec));
        //auto re_gpu0 = std::accumulate(c_vec.begin(), c_vec.end(), decltype(c_vec)::value_type(0));

        std::vector<cl::Event> passes;
        cl_uint curr = static_cast<cl_uint>(N);
        while ( curr > 1 )
        {
            passes.push_back(
                reduce(
                    cl::EnqueueArgs{
                        queue,          //CommandQueue
                        passes,         //events
                        global(curr),   //NDRange global
                        wgs             //NDRange local
                    },
                    c_buf,
                    red_buf,
                    cl::Local(factor * sizeof(cl_float)),
                    curr,
                    zero_elem
                ) 
            );
            curr = static_cast<cl_uint>(new_size(curr));
            if (curr > 1) std::swap(c_buf, red_buf);
        }
        for (auto& pass : passes) pass.wait();
        auto end_gpu = tmark();

        // (Blocking) fetch of results
        cl_float re_gpu;
        cl::copy(queue, red_buf, &re_gpu, &re_gpu + 1);
        cl::finish();
        //--------------------------------------------------------------------------------------------

        //naive implementation
        auto start_naiv = tmark();
        auto re_cpu = cpu_scalar_prod_naive(a_vec, b_vec, N);
        auto end_naiv = tmark();

        //parallel implementation
        auto start_par = tmark();
        auto re_cpu_par = cpu_scalar_prod_parallel(a_vec, b_vec, N);
        auto end_par = tmark();

        
        //Reference
        auto start_ref = tmark();
        auto re_ref = std::inner_product(std::begin(a_vec), std::end(a_vec), std::begin(b_vec), 0.0);
        auto end_ref = tmark();

        //Results
        std::cout.precision(10);

        auto re_err = std::abs((re_ref - re_gpu) / re_ref);
        
        if( re_err < 2e-4 )
        {
            std::cout << "Validation success.\n";
            std::cout << "Result: " << re_ref << std::endl;
            std::cout << "Relative error between CPU & GPU is: " << re_err << std::endl;
            std::cout << "Device execution took:        " << delta_time(start_gpu,end_gpu)   << " ms" << std::endl;
            std::cout << "Ref. host execution took:     " << delta_time(start_ref,end_ref) << " ms" << std::endl;
            std::cout << "Naive host execution took:    " << delta_time(start_naiv,end_naiv) << " ms" << std::endl;
            std::cout << "Paralell host execution took: " << delta_time(start_par,end_par)   << " ms" << std::endl;
        }
        else
        {
            std::cout << "Mismatch in CPU and GPU result.\n";
            std::cout << "Reference:           " << re_ref << std::endl;
            std::cout << "Result of GPU:       " << re_gpu << std::endl;
            std::cout << "Result of naive:     " << re_cpu << std::endl;
            std::cout << "Result of parallel:  " << re_cpu_par << std::endl;
            std::cout << "Relative error between CPU & GPU is: " << re_err << std::endl;
            std::cout << "Device execution took:        " << delta_time(start_gpu,end_gpu)   << " ms" << std::endl;
            std::cout << "Ref. host execution took:     " << delta_time(start_ref,end_ref) << " ms" << std::endl;
            std::cout << "Naive host execution took:    " << delta_time(start_naiv,end_naiv) << " ms" << std::endl;
            std::cout << "Parallel host execution took: " << delta_time(start_par,end_par)   << " ms" << std::endl;
        }
    }
    catch (cl::BuildError& error) // If kernel failed to build
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}