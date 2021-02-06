#pragma once
#include <iostream>
#include <string>
#include <string_view>
#include "Digits.hpp"
#include "util.hpp"

namespace mmath {

class Int {
private:
	using Sign = bool;
	static constexpr Sign PLUS = true;
	static constexpr Sign MINUS = false;

	// log2(digitsの基数)、つまり基数は2の何乗かを示す
	// 16進数への変換を考えて、LOG_RADIX % 4 = 0 である必要あり
	static constexpr i32 LOG_RADIX = 20;

	mmath::Digits digits;
	Sign sign;

public:
	Int();
	Int(const mmath::Int &) = default;
	Int(mmath::Int &&) = default;
	Int(const Digits &x);

	// 16進文字列を受け取って数値に変換
	Int(std::string_view x);
};

}
