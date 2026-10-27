// Weight-streaming GEMM for one AIE2P core of a row chain: bfp16 (EBS8)
// operands, fp32 accumulators, partial sums passed west to east over the
// cascade.
//
// A row of 8 cores computes C[MT rows, N] = A[MT, K] x W[N, K]^T with the
// reduction split 8 ways: core c holds the K slice [c*K/8, (c+1)*K/8) of its
// MT activation rows (loaded once per GEMM, in chunks of KS) and receives that
// slice of W one output group (NG columns) at a time. Per group it starts from
// zero (head) or from the partial sums of the core to its west (cascade), adds
// its slice, and passes the sums east (or, at the tail, writes them out).
//
// Layouts (bytes; a block is 8x8 bfp16 = 8 rows of [exponent, 8 mantissas]):
//   A chunk  [MT/16 row pairs][KS/8 k blocks][2][72]   rows m, cols k
//   abuf     [MT/16][KBMAX][2][72]  the core's whole K slice, chunk ch at KS/8*ch
//   B object [NG/16 col pairs][KS/8][2][72]            rows n, cols k
//   acc      [MT/8][NG/8][64] fp32                     rows m, cols n
//
// MT, NG set the geometry; one build serves every K <= 8 * KBMAX * 8 and N.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MT
#define MT 32
#endif
#ifndef NG
#define NG 32
#endif
#define KS 128   // k per chunk
#define KBMAX 48 // k blocks of one core's slice at the largest K (3072)

static_assert(MT % 16 == 0 && NG % 16 == 0, "row and column pairs");

constexpr int BLK = 72;                    // one 8x8 bfp16 block
constexpr int PAIR = 2 * BLK;              // two blocks, one k step of a pair
constexpr int CHUNK_PAIR = KS / 8 * PAIR;  // one row (or column) pair of a chunk
constexpr int ABUF_PAIR = KBMAX * PAIR;    // one row pair of the K slice
constexpr int ACC_BLOCKS = (MT / 8) * (NG / 8);

using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using BS = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;

static inline ACC load_acc(const float *p) {
  ACC a;
  a.from_vector(aie::load_v<64>(p));
  return a;
}

static inline ACC mac(ACC c, BV a, BV b) {
  return ACC(::mac_8x8_8x8T(a, b, c));
}

extern "C" {

// Chunk ch of this core's A slice into abuf.
void ws_load_a(const uint8_t *chunk, uint8_t *abuf, int32_t ch) {
  for (int p = 0; p < MT / 16; p++) {
    const v64int8 *s = (const v64int8 *)(chunk + p * CHUNK_PAIR);
    v64int8 *d = (v64int8 *)(abuf + p * ABUF_PAIR + ch * CHUNK_PAIR);
    for (int i = 0; i < CHUNK_PAIR / 64; i++)
      d[i] = s[i];
  }
}

void ws_zero(float *acc) {
  for (int i = 0; i < ACC_BLOCKS; i++)
    aie::store_v(acc + 64 * i, aie::zeros<float, 64>());
}

// acc += A[:, chunk ch] x B_object^T
void ws_mac(const uint8_t *abuf, const uint8_t *b, float *acc, int32_t ch) {
  for (int p = 0; p < MT / 16; p++) {
    for (int q = 0; q < NG / 16; q++) {
      float *c00 = acc + ((2 * p) * (NG / 8) + 2 * q) * 64;
      float *c01 = c00 + 64;
      float *c10 = c00 + (NG / 8) * 64;
      float *c11 = c10 + 64;
      ACC a00 = load_acc(c00), a01 = load_acc(c01);
      ACC a10 = load_acc(c10), a11 = load_acc(c11);
      BS as((const bfp16ebs8 *)(abuf + p * ABUF_PAIR + ch * CHUNK_PAIR));
      BS bs((const bfp16ebs8 *)(b + q * CHUNK_PAIR));
      for (int kb = 0; kb < KS / 8; kb++) {
        BV x0, x1, y0, y1;
        as >> x0 >> x1;
        bs >> y0 >> y1;
        a00 = mac(a00, x0, y0);
        a01 = mac(a01, x0, y1);
        a10 = mac(a10, x1, y0);
        a11 = mac(a11, x1, y1);
      }
      aie::store_v(c00, a00.to_vector<float>());
      aie::store_v(c01, a01.to_vector<float>());
      aie::store_v(c10, a10.to_vector<float>());
      aie::store_v(c11, a11.to_vector<float>());
    }
  }
}

// The group's partial sums to the core east of this one.
void ws_put(const float *acc) {
  for (int i = 0; i < ACC_BLOCKS * 4; i++)
    put_mcd(*(const v16float *)(acc + 16 * i));
}

// The partial sums of the core west of this one, as this core's start.
void ws_get(float *acc) {
  for (int i = 0; i < ACC_BLOCKS * 4; i++)
    *(v16float *)(acc + 16 * i) = (v16float)get_scd_v16accfloat(1);
}

// Tail: the group's sums out, fp32, in the acc layout.
void ws_out_f32(const float *acc, float *out) {
  for (int i = 0; i < ACC_BLOCKS; i++)
    aie::store_v(out + 64 * i, aie::load_v<64>(acc + 64 * i));
}

} // extern "C"
