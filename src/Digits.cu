#include "Digits.cuh"

__host__ __device__ 
void mmath::Digits_utils::small_hex_str_to_digit_type(const char *st, i32 len, digit_type *res) {
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
void mmath::Digits_utils::decode_hex_for_digits(const char *st, size_t len, i32 size, digit_type *data, size_t N) {
	assert(N != 0);
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i == N - 1) {
		mmath::Digits_utils::small_hex_str_to_digit_type(st, len - i * size, &(data[i]));
	} else if(i < N - 1) {
		mmath::Digits_utils::small_hex_str_to_digit_type(&(st[len - size * (i + 1)]), size, &(data[i]));
	}
}

__global__
void mmath::Digits_utils::sum_for_look_ahead(digit_type *a, const digit_type *b, size_t len, i32 LOG_RADIX, char *ps, char *gs) {
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
void mmath::Digits_utils::sum_for_look_ahead_carry(digit_type *data, char *carrys, size_t len, i32 LOG_RADIX, char *c) {
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < len) {
		data[i] = (data[i] + carrys[i - 1]) & (Digits::RADIX - 1);
	}
	if(i == len - 1) *c = carrys[i];
}

__global__
void mmath::Digits_utils::not_for_complement(digit_type *data, size_t len, i32 LOG_RADIX) {
	i64 i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < len) data[i] = (~data[i]) & (Digits::RADIX - 1);
}

__global__
void mmath::Digits_utils::assign_eq_index_else_n(const digit_type *x, digit_type r, digit_type *p, digit_type n, size_t len) {
	i32 i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= len) return;

	if(x[i] == r) p[i] = i;
	else p[i] = n;
}

__global__
void mmath::Digits_utils::assign_eq_index_else_n(const digit_type *x, digit_type r, digit_type *p, digit_type n, size_t len, digit_type l) {
	i32 i = blockDim.x * blockIdx.x + threadIdx.x + l;
	if(i >= len) return;

	if(x[i] < r) p[i] = i;
	else p[i] = n;
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

#if MMATH_DIGITS_ALIGN_SEQUENTIAL
void mmath::Digits::align() {
	for(size_t i = 0; i < size() - 1; i++) {
		if(data[i] >= RADIX) {
			data[i + 1] += data[i] >> LOG_RADIX;
			data[i] &= (RADIX - 1);
		}
	}

	if(msd() >= RADIX) {
		push_msd(msd() >> LOG_RADIX);
		data[size() - 2] &= (RADIX - 1);
	}
}
#endif

#if MMATH_DIGITS_ALIGN_PARALLEL
void mmath::Digits::align() {
	size_t n = size();
	thrust::device_vector<digit_type> c(n);

	while(RADIX < *thrust::max_element(data.begin(), data.end() - 1)) {
		c[0] = 0;
		thrust::transform(data.begin(), data.end() - 1, c.begin() + 1, mmath::Digits_utils::divide_radix());
		thrust::transform(data.begin(), data.end() - 1, c.begin(), data.begin(), mmath::Digits_utils::mod_radix_add());
		data[n - 1] += c[n - 1];
	}

	i32 dB = MAX_X_THREAD_SIZE;
	i32 dG1 = (n >> LOG_MAX_X_THREAD_SIZE) + 1;
	i32 dG2;
	thrust::device_vector<digit_type> p(n);
	digit_type l, m;
	while(RADIX == *thrust::max_element(data.begin(), data.end() - 1)) {
		mmath::Digits_utils::assign_eq_index_else_n<<<dG1, dB>>>(thrust::raw_pointer_cast(data.data()), RADIX, thrust::raw_pointer_cast(p.data()), n - 1, n);
		l = *thrust::min_element(p.begin(), p.end());

		dG2 = ((n - l - 1) >> LOG_MAX_X_THREAD_SIZE) + 1;
		mmath::Digits_utils::assign_eq_index_else_n<<<dG2, dB>>>(thrust::raw_pointer_cast(data.data()), RADIX - 1, thrust::raw_pointer_cast(p.data()), n - 1, n, l + 1);

		m = *thrust::min_element(p.begin() + l + 1, p.end());
		data[l] -= RADIX;

		if(l + 2 <= m) thrust::transform(data.begin() + l + 1, data.begin() + m - 1, data.begin() + l + 1, mmath::Digits_utils::sub_radix_1());
		data[m]++;
	}

	if(data[n - 1] >= RADIX) {
		data.push_back(data[n - 1] >> LOG_RADIX);
		data[n - 1] &= (RADIX - 1);
	}
}
#endif

// host
// sequential algorithm
#if MMATH_DIGITS_ADD_SEQUENTIAL
void mmath::Digits::add(const mmath::Digits &x) {
	// if(x == 0) return;
	if(size() < x.size()) data.resize(x.size());

	digit_type buf, carry = 0;
	digit_type rd;
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

// host
// sequential and naive
#if MMATH_DIGITS_MUL_SEQ_SCHOOL
void mmath::Digits::mul(const mmath::Digits &x) {
	mmath::Digits a(*this);
	to_zero();
	
	for(size_t i = 0; i < x.size(); i++) {
		mmath::Digits tmp(i, 0);
		digit_type carry = 0;
		for(size_t j = 0; j < a.size(); j++) {
			tmp.push_msd((a.data[j] * x.data[i] + carry) % RADIX);
			carry = (a.data[j] * x.data[i] + carry) / RADIX;
		}
		if(carry != 0) tmp.push_msd(carry);
		add(tmp);
	}
}
#endif

// host
// sequential ntt
#if MMATH_DIGITS_MUL_SEQ_NTT
void mmath::Digits::mul(const mmath::Digits &x) {
	constexpr digit_type MOD = 3221225473;	// 3 * 2^30 + 1
	constexpr digit_type g = 5;
	size_t s = size() + x.size() - 1;
	size_t N = 1;
	while(N < s) N <<= 1;

	mmath::Digits x_(x);
	data.resize(N);
	x_.data.resize(N);

	mmath::NTT::ntt_cpu<digit_type, MOD, g>(data);
	mmath::NTT::ntt_cpu<digit_type, MOD, g>(x_.data);

	thrust::transform(data.begin(), data.end(), x_.data.begin(), data.begin(), mmath::NTT::mul_op<digit_type, MOD>());

	mmath::NTT::ntt_cpu<digit_type, MOD, g>(data, true);

	normalize();
	align();
}
#endif

// device
// parallel ntt
#if MMATH_DIGITS_MUL_PARALLEL_NTT
void mmath::Digits::mul(const mmath::Digits &x) {
	constexpr digit_type MOD = 3221225473;	// 3 * 2^30 + 1
	constexpr digit_type g = 5;
	size_t s = size() + x.size() - 1;
	size_t n = 1;
	while(n < s) n <<= 1;
	size_t nh = (n >> 1);

	mmath::Digits x_(x);
	data.resize(n);
	x_.data.resize(n);

	// rotation table
	thrust::device_vector<digit_type> ws(nh);
	i32 dB = MAX_X_THREAD_SIZE;
	i32 dG = (nh >> LOG_MAX_X_THREAD_SIZE) + 1;
	mmath::NTT::gen_rotation_table<digit_type, MOD, g><<<dG, dB>>>(thrust::raw_pointer_cast(ws.data()), nh);

	mmath::NTT::ntt_no_bitrev<digit_type, MOD, g>(data, ws);
	mmath::NTT::ntt_no_bitrev<digit_type, MOD, g>(x_.data, ws);

	thrust::transform(data.begin(), data.end(), x_.data.begin(), data.begin(), mmath::NTT::mul_op<digit_type, MOD>());

	mmath::NTT::ntt_no_bitrev<digit_type, MOD, g>(data, ws, true);

	normalize();
	align();
}
#endif
