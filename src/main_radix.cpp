#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 10;
const int benchmarkingItersCPU = 1;
const unsigned int n = 32 * 1024 * 1024;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::vector<unsigned int> computeCPU(const std::vector<unsigned int> &as)
{
    std::vector<unsigned int> cpu_sorted;

    timer t;
    for (int iter = 0; iter < benchmarkingItersCPU; ++iter) {
        cpu_sorted = as;
        t.restart();
        std::sort(cpu_sorted.begin(), cpu_sorted.end());
        t.nextLap();
    }
    std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

    return cpu_sorted;
}

void calc_prefix_sum(
    ocl::Kernel& prefix_sum, gpu::gpu_mem_32u& a_gpu,
    unsigned int n, unsigned int workgroup_size
) {
    unsigned int offset;
    for (offset = 1; offset < n; offset <<= 1) {
        prefix_sum.exec(
            gpu::WorkSize(workgroup_size, n / (offset * 2)),
            a_gpu, a_gpu,
            n, offset * 2 - 1, offset * 2, offset
        );
    }

    offset >>= 1;
    for (offset >>= 1; offset > 0; offset >>= 1) {
        prefix_sum.exec(
            gpu::WorkSize(workgroup_size, n / (offset * 2)),
            a_gpu, a_gpu,
            n, offset * 3 - 1, offset * 2, offset
        );
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const std::vector<unsigned int> cpu_reference = computeCPU(as);

    ocl::Kernel set_zeros(radix_kernel, radix_kernel_length, "set_zeros");
    set_zeros.compile();
    ocl::Kernel matrix_transpose_kernel(radix_kernel, radix_kernel_length, "matrix_transpose_local_memory_kernel");
    matrix_transpose_kernel.compile();
    ocl::Kernel prefix_sum_kernel(radix_kernel, radix_kernel_length, "prefix_sum_kernel");
    prefix_sum_kernel.compile();
    ocl::Kernel calc_cnt_kernel(radix_kernel, radix_kernel_length, "calc_cnt_kernel");
    calc_cnt_kernel.compile();
    ocl::Kernel radix_sort_kernel(radix_kernel, radix_kernel_length, "radix_sort_kernel");
    radix_sort_kernel.compile();

    const unsigned int workgroup_size = 128;
    const unsigned int transpose_tile_size = 16;
    const unsigned int bits_num = 4;
    const unsigned int counter_size = 1 << bits_num;
    const unsigned int workgroups_num = (n + workgroup_size - 1) / workgroup_size;
    const unsigned int total_counters_size = workgroups_num * counter_size;

    gpu::gpu_mem_32u a_gpu, b_gpu, cnt_gpu, cnt_t_gpu;
    a_gpu.resizeN(n);
    b_gpu.resizeN(n);
    cnt_gpu.resizeN(total_counters_size);
    cnt_t_gpu.resizeN(total_counters_size);

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            a_gpu.writeN(as.data(), as.size());

            t.restart();

            for (int shift = 0; shift < 32; shift += bits_num) {
                set_zeros.exec(
                    gpu::WorkSize(workgroup_size, total_counters_size),
                    cnt_gpu, total_counters_size
                );

                calc_cnt_kernel.exec(
                    gpu::WorkSize(workgroup_size, n),
                    a_gpu, cnt_gpu,
                    n, bits_num, shift
                );

                matrix_transpose_kernel.exec(
                    gpu::WorkSize(transpose_tile_size, transpose_tile_size, counter_size, workgroups_num),
                    cnt_gpu, cnt_t_gpu,
                    workgroups_num, counter_size
                );

                calc_prefix_sum(prefix_sum_kernel, cnt_t_gpu, total_counters_size, workgroup_size);

                radix_sort_kernel.exec(
                    gpu::WorkSize(workgroup_size, n),
                    a_gpu, b_gpu, cnt_t_gpu,
                    n, bits_num, shift, workgroups_num
                );

                std::swap(a_gpu, b_gpu);
            }

            t.nextLap();
        }
        t.stop();

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    a_gpu.readN(as.data(), n);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_reference[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
