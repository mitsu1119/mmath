#include "Digits.cuh"

__host__ __device__ 
void mmath::Digits_utils::small_hex_str_to_i64(const char *st, i32 len, i64 *res) {
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
void mmath::Digits_utils::decode_hex_for_digits(const char *st, size_t len, i32 size, i64 *data, size_t N) {
	assert(N != 0);
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i == N - 1) {
		mmath::Digits_utils::small_hex_str_to_i64(st, len - i * size, &(data[i]));
	} else if(i < N - 1) {
		mmath::Digits_utils::small_hex_str_to_i64(&(st[len - size * (i + 1)]), size, &(data[i]));
	}
}

__global__
void mmath::Digits_utils::sum_for_look_ahead(i64 *a, const i64 *b, size_t len, i32 LOG_RADIX, char *ps, char *gs) {
	assert(len != 0);
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < len) {
		gs[i] = (a[i] + b[i]) >> LOG_RADIX;
		a[i] = (a[i] + b[i]) & (Digits::RADIX - 1);
		ps[i] = (char)(a[i] == (Digits::RADIX - 1));
	}
}

__global__
void mmath::Digits_utils::carrys_for_look_ahead(char *ps, char *gs, size_t k, size_t len) {
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;

	char g, p;
	if(i < len && i >= k) {
		g = gs[i] | (ps[i] & gs[i - k]);
		p = (ps[i] & ps[i - k]);
	}

	__syncthreads();
	if(i < len && i >= k) {
		gs[i] = g;
		ps[i] = p;
	}
}

__global__
void mmath::Digits_utils::sum_for_look_ahead_carry(i64 *data, char *carrys, size_t len, i32 LOG_RADIX, char *c) {
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < len) {
		data[i] = (data[i] + carrys[i - 1]) & (Digits::RADIX - 1);
	}
	if(i == len - 1) *c = carrys[i];
}

__global__
void mmath::Digits_utils::not_for_complement(i64 *data, size_t len, i32 LOG_RADIX) {
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < len) data[i] = (~data[i]) & (Digits::RADIX - 1);
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

	mmath::Digits_utils::decode_hex_for_digits<<<blocks_per_grid, threads_per_block>>>(d_st, len_st, mmath::Digits::LOG_16_RADIX, thrust::raw_pointer_cast(data.data()), data.size());
	cudaDeviceSynchronize();

	cudaFree(d_st);
}

// host
// sequential algorithm
#if MMATH_DIGITS_ADD_SEQUENTIAL
void mmath::Digits::add(const mmath::Digits &x) {
	// if(x == 0) return;
	if(size() < x.size()) data.resize(x.size());

	i64 buf, carry = 0;
	i64 rd;
	for(size_t i = 0; i < size(); i++) {
		rd = (i < x.size()) ? x.at(i) : 0;
		buf = at(i) + rd + carry;
		carry = buf >> mmath::Digits::LOG_RADIX;
		data[i] = buf & ((1 << mmath::Digits::LOG_RADIX) - 1);
	}

	if(carry != 0) push_msd(carry);
	normalize();
}
#endif

// device
// carry look-ahead
#if MMATH_DIGITS_ADD_PARALLEL
void mmath::Digits::add(const mmath::Digits &xx) {
	// if(x == 0) return;

	mmath::Digits x(xx);
	if(size() < x.size()) data.resize(x.size());
	x.data.resize(size());

	size_t len = size();
	char *ps;
	char *gs;

	cudaMalloc(&ps, len);
	cudaMalloc(&gs, len);
	
	// look-ahead sum
	i32 threads_per_block = 512;
	i32 blocks_per_grid;
	if(len & (512 - 1) == 0) blocks_per_grid = len >> 9;
	else blocks_per_grid = (len >> 9) + 1;

	mmath::Digits_utils::sum_for_look_ahead<<<blocks_per_grid, threads_per_block>>>(thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(x.data.data()), len, mmath::Digits::LOG_RADIX, ps, gs);
	cudaDeviceSynchronize();

	for(size_t k = 1; k < len; k <<= 1) {
		mmath::Digits_utils::carrys_for_look_ahead<<<blocks_per_grid, threads_per_block>>>(ps, gs, k, len);
	}

	char c;
	char *d_c;
	cudaMalloc(&d_c, 1);

	mmath::Digits_utils::sum_for_look_ahead_carry<<<blocks_per_grid, threads_per_block>>>(thrust::raw_pointer_cast(data.data()), gs, len, mmath::Digits::LOG_RADIX, d_c);
	cudaDeviceSynchronize();

	cudaMemcpy(&c, d_c, 1, cudaMemcpyDeviceToHost);
	cudaFree(d_c);

	if(c != 0) push_msd(c);

	cudaFree(gs);
	cudaFree(ps);
}
#endif

void mmath::Digits::sub(const mmath::Digits &x) {
	/*
	if(x == *this) {
		to_zero();
		return;
	}
	*/

	mmath::Digits s(x);
	size_t len = size();
	if(s.size() < len) s.data.resize(len);	

	i32 threads_per_block = 512;
	i32 blocks_per_grid;
	if(len & (512 - 1) == 0) blocks_per_grid = len >> 9;
	else blocks_per_grid = (len >> 9) + 1;

	mmath::Digits_utils::not_for_complement<<<blocks_per_grid, threads_per_block>>>(thrust::raw_pointer_cast(s.data.data()), len, mmath::Digits::LOG_RADIX);
	cudaDeviceSynchronize();

	add(s);	// contain normalize()
	increment();
	pop_msd();
}

void mmath::Digits::increment() {
	size_t idx = first_non_radixmax_index();

	if(idx >= data.size()) {
		thrust::fill(data.begin(), data.end(), 0);
		push_msd(1);
	} else {
		thrust::fill(data.begin(), data.begin() + idx, 0);
		data[idx] += 1;
	}
}
