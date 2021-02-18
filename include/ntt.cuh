#pragma once
#include <thrust/device_vector.h>
#include "util.cuh"

namespace mmath {

namespace NTT {

template <typename T, T MOD>
inline T add(const T &a, const T &b) {
	return (a + b < MOD) ? a + b : a + b - MOD;
}

template <typename T, T MOD>
inline T sub(const T &a, const T &b) {
	return (a > b) ? a - b : (a + (MOD - b)) % MOD;
}

template <typename T, T MOD>
inline T mul(const T &a, const T &b) {
	return (a * b) % MOD;
}

template <typename T, T MOD>
inline T pow(T x, T n) {
	T res = 1;
	for(; n > 0; n >>= 1, x = mmath::NTT::mul<T, MOD>(x, x)) {
		if(n & 1 == 1) res = mmath::NTT::mul<T, MOD>(res, x);
	}

	return res;
}

template <typename T, T MOD>
inline T modinv(const T &x) {
	return mmath::NTT::pow<T, MOD>(x, MOD - 2);
}

template <typename T, T MOD, T primitive_root>
void ntt(thrust::device_vector<T> &f, bool rev = false) {
	size_t n = f.size();
	if(n == 1) return;
	size_t nh = n >> 1;

	size_t i, j, k, m, mh, ts;
	T tmp;

	// rotation table
	std::vector<T> ws(nh);
	tmp = 1;
	T root;
	if(rev) root = mmath::NTT::modinv<T, MOD>(mmath::NTT::pow<T, MOD>(primitive_root, (MOD - 1) / n));
	else root = mmath::NTT::pow<T, MOD>(primitive_root, (MOD - 1) / n);
	for(i = 0; i < nh; i++) {
		ws[i] = tmp;
		tmp = mmath::NTT::mul<T, MOD>(tmp, root);
	}

	// bit rev
	i = 0;	
	for(j = 1; j < n - 1; j++) {
		for(k = nh; k > (i ^= k); k >>= 1);
		if(j < i) {
			// thrust::swap
			tmp = f[i];
			f[i] = f[j];
			f[j] = tmp;
		}
	}

	T l, r;
	for(m = 2; m <= n; m <<= 1) {
		mh = m >> 1;
		ts = n / m;
		for(i = 0; i < n; i += m) {
			k = 0;
			for(j = i; j < i + mh; j++) {
				l = f[j];
				r = mmath::NTT::mul<T, MOD>(f[j + mh], ws[k]);
				f[j] = mmath::NTT::add<T, MOD>(l, r);
				f[j + mh] = mmath::NTT::sub<T, MOD>(l, r);
				k += ts;
			}
		}
		if(m == n) break;
	}

	if(rev) {
		T inv = mmath::NTT::modinv<T, MOD>(n);
		for(i = 0; i < n; i++) {
			f[i] = mmath::NTT::mul<T, MOD>(f[i], inv);
		}
	}
}

}

}

