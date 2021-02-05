#include <iostream>
#include <time.h>
#include <unistd.h>
#include "Int.hpp"

using namespace mmath;
int main() {
	// 計測用
	float elapsed;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	Digits d(1ul << 10, 50);

	cudaEventRecord(start, 0);

	d.to_zero();
	d.print();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << "elapsed: " << elapsed << " [ms]" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
