#include "Int.cuh"

mmath::Int::Int(): sign(PLUS) {
	digits.to_zero();
}

mmath::Int::Int(const Digits &digits): digits(digits), sign(PLUS) {
}


mmath::Int::Int(const std::string &x): sign(PLUS) {
	digits.from_hex(x.c_str(), x.size());
}

void mmath::Int::print_hex(std::ostream &os) const {
	os << "0x" << std::hex << digits.msd() << std::setfill('0') << std::right << std::setw(LOG_16_RADIX);

	size_t len = digits.size();
	for(size_t i = 1; i < len; i++) os << digits.at(len - i - 1);
	os << std::endl;
}

mmath::Int mmath::Int::abs() const {
	return mmath::Int(digits);
}
