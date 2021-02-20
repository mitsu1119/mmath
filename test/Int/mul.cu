#include <iostream>
#include <time.h>
#include <unistd.h>
#include "Int.cuh"

using namespace mmath;
int main() {

	Int c("ffffffffabcdef123456789ffffffffffabcdef1234567890abcdef1234567890deadbeefdeadbeefabcdef1234567890abcdef1234567890deadbeefdeadbeefabcdef1234567890");
	Int d("ffffffffabcdef123456789ffffffffffabcdef1234567890abcdef1234567890deadbeefdeadbeefabcdef1234567890abcdef1234567890deadbeefdeadbeefabcdef1234567890");

	// 計測用(variable)
	float elapsed;
	cudaEvent_t start, stop;

	// 計測用(start)
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	c.mul(d);

	// 計測用(end)
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << "elapsed: " << elapsed << " [ms]" << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << c << std::endl;

	return 0;
}
