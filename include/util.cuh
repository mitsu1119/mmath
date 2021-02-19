#pragma once
#include <cstdint>

typedef uint_fast64_t u64;
typedef uint_fast32_t u32;
typedef int_fast64_t i64;
typedef int_fast32_t i32;

#define MAX_BLOCK_SIZE 512
#define MAX_X_THREAD_SIZE 512
#define MAX_Y_THREAD_SIZE 512
#define MAX_Z_THREAD_SIZE 64

// BLOCK_SIZE_DIV1 * BLOCK_SIZE_DIV2 == MAX_BLOCK_SIZE
#define BLOCK_SIZE_DIV1 32
#define BLOCK_SIZE_DIV2 16	
