// Weight-stationary implicit-GEMM 3x3 conv block for AIE2P, int8 x int8 -> int32 accumulators.
//
// Layouts in local memory (all 8-channel interleaved, so one 8x8 int8 operand is 64 contiguous bytes):
//   X[kx][cg][p][8]   input tile, kx = 0..2 copies shifted by kx pixels (keeps every tap load aligned),
//                     cg = input-channel group of 8, p = pixel on the WP-wide padded grid.
//   W[tap][cg][og][8][8]  weights of this core's channel slice, og = output-channel group of 8.
//   O[og][p][8]       output (int8 after shift-round-saturate), same layout as X for the next layer.
// One block is 16 pixels x 16 output channels (2x2 accumulators of 8x8 int32), reduced over
// 9 taps x CG channel groups without touching local memory for the accumulators.
#include <aie_api/aie.hpp>

#define UNROLL_FULL _Pragma("clang loop unroll(full)")
#define PRAGMA_STR(x) #x
#define MIN_ITERS(x) _Pragma(PRAGMA_STR(clang loop min_iteration_count(x)))

// Weights in consumption order: Wc[og/2][cg][tap][2][64] (the two 8x8 blocks of an output-channel
// pair side by side), so one pointer walks them; the input pointer walks kx, ky, cg with the 3D
// address generator. Chained addresses keep the scheduler from hoisting every load up front.
template <int CG, int WP, int P, int OG, int NPIX>
static inline void conv3x3_ws(const int8 *__restrict X, const int8 *__restrict Wc, int8 *__restrict O,
                              int shift) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int XS = CG * P * 8;            // bytes per kx copy
  // One flat loop over (output-channel pair, 16-pixel block): the weight pointer then changes with
  // the loop variable, so LLVM does not hoist all CG*9 weight loads (9 KB) out of it onto the stack.
  constexpr int NB = NPIX / 16;
  const int8 *__restrict wog = Wc;
  int pb = 0;
  int og = 0;
  MIN_ITERS(2)
  for (int it = 0; it < (OG / 2) * NB; ++it) {
    const int p0 = pb * 16;
    dims_3d_t ad = dims_3d_from_steps(3, XS, 3, WP * 8, P * 8);
    const int8 *__restrict xa = X + p0 * 8;
    const int8 *__restrict wb = wog;
    MMUL C00, C01, C10, C11;
    auto step = [&](bool first) [[gnu::always_inline]] {
      auto A0 = aie::load_v<64>(xa);
      auto A1 = aie::load_v<64>(xa + 64);
      xa = add_3d_byte(xa, ad);
      auto B0 = aie::load_v<64>(wb);
      auto B1 = aie::load_v<64>(wb + 64);
      wb += 128;
      if (first) {
        C00.mul(A0, B0); C01.mul(A0, B1); C10.mul(A1, B0); C11.mul(A1, B1);
      } else {
        C00.mac(A0, B0); C01.mac(A0, B1); C10.mac(A1, B0); C11.mac(A1, B1);
      }
    };
    step(true);
    UNROLL_FULL
    for (int i = 1; i < CG * 9; ++i)
      step(false);
    int8 *o0 = O + (og * P + p0) * 8;
    int8 *o1 = O + ((og + 1) * P + p0) * 8;
    aie::store_v(o0, C00.template to_vector<int8>(shift));
    aie::store_v(o0 + 64, C10.template to_vector<int8>(shift));
    aie::store_v(o1, C01.template to_vector<int8>(shift));
    aie::store_v(o1 + 64, C11.template to_vector<int8>(shift));
    // next block: next 16 pixels, or the next output-channel pair
    const bool wrap = pb == NB - 1;
    pb = wrap ? 0 : pb + 1;
    og = wrap ? og + 2 : og;
    wog = wrap ? wog + CG * 9 * 128 : wog;
  }
}

extern "C" {
// layer-sized example: 64 input channels (8 groups), 64 output channels, 16x16 output pixels on a
// 24-wide padded grid (WP multiple of 8 keeps ky*WP aligned)
void conv3x3_ws_c64(const int8 *X, const int8 *W, int8 *O, int shift) {
  conv3x3_ws<8, 24, 24 * 18, 8, 24 * 16>(X, W, O, shift);
}
}
