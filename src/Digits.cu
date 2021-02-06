#include "Digits.cuh"

__host__ __device__ 
void mmath::small_hex_str_to_i64(const char *st, i32 len, i64 *res) {
	*res = 0;
	for(i32 i = 0; i < len; i++) {
		char c = st[i];
		if('0' <= c && c <= '9') *res += c - '0';
		else if('A' <= c && c <= 'F') *res += c - 'A' + 10;
		else if('a' <= c && c <= 'f') *res += c - 'a' + 10;
		else break;

		*res <<= 4;
	}
	*res >>= 4;
}

__global__
void mmath::decode_hex_for_digits(const char *st, size_t len, i32 size, i64 *data, size_t N) {
	assert(N != 0);
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i == N - 1) {
		mmath::small_hex_str_to_i64(st, len - i * size, &(data[i]));
	} else if(i < N - 1) {
		mmath::small_hex_str_to_i64(&(st[len - size * (i + 1)]), size, &(data[i]));
	}
}

void mmath::Digits::print(bool hex) const {
	if(hex) std::cout << std::hex;
	for(auto i: data) std::cout << i << " ";
	std::cout << std::endl;
}

void mmath::Digits::from_hex(const char *st, size_t len_st) {
	if(len_st == 0) {
		to_zero();
		return;
	}

	if(len_st % mmath::Digits::LOG_16_RADIX == 0) {
		data.resize(len_st / mmath::Digits::LOG_16_RADIX);
	} else {
		data.resize(len_st / mmath::Digits::LOG_16_RADIX + 1);
	}

	i32 threads_per_block = 512;
	i32 blocks_per_grid;
	if(len_st & (512 - 1) == 0) blocks_per_grid = len_st >> 9;
	else blocks_per_grid = (len_st >> 9) + 1;

	char *d_st;
	cudaMalloc(&d_st, len_st);
	cudaMemcpy(d_st, st,len_st, cudaMemcpyHostToDevice);

	mmath::decode_hex_for_digits<<<blocks_per_grid, threads_per_block>>>(d_st, len_st, mmath::Digits::LOG_16_RADIX, thrust::raw_pointer_cast(data.data()), data.size());

	cudaFree(d_st);
}
