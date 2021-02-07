#pragma once
#include <iostream>
#include <iomanip>
#include "Digits.cuh"
#include "util.cuh"

namespace mmath {

class Int {
private:
	using Sign = bool;
	static constexpr Sign PLUS = true;
	static constexpr Sign MINUS = false;

	static constexpr i32 LOG_RADIX = mmath::Digits::LOG_RADIX;
	static constexpr i32 LOG_16_RADIX = mmath::Digits::LOG_16_RADIX;

	mmath::Digits digits;
	Sign sign;

	void print_hex(std::ostream &os) const;

public:
	Int();
	Int(const mmath::Int &) = default;
	Int(mmath::Int &&) = default;
	Int(const Digits &x);

	// 16進文字列を受け取って数値に変換
	Int(const std::string &x);

	mmath::Int abs() const;

	void add(const mmath::Int &x);
	void sub(const mmath::Int &x);

	// operators
	friend std::ostream &operator<<(std::ostream &os, const mmath::Int &x) {
		x.print_hex(os);
		return os;
	}
};

inline void mmath::Int::add(const mmath::Int &x) {
	digits.add(x.digits);
}

inline void mmath::Int::sub(const mmath::Int &x) {
	digits.sub(x.digits);
}

}
