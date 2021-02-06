#include <iostream>
#include <time.h>
#include <unistd.h>
#include "Int.cuh"

using namespace mmath;
int main() {
	// 計測用(start)
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	Int c("deadbeefabcdef1234567890");
	std::cout << c << std::endl;

	// 計測用(end)
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << "elapsed: " << elapsed << " [ms]" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
