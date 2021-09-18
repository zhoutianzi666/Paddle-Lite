/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "lite/backends/x86/math/conv3x3s1_direct.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/core/context.h"
#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

struct  jit_3x3s1_param
{
  const float* input_address;
  const float* kernel_address;
  float* output_address;
};

void conv_direct_3x3s1::generate(int ic,
                  int ih,
                  int iw,
                  int oc,
                  int oc_expand,
                  int oh,
                  int ow,
                  int ph,
                  int pw)
{

#ifdef __AVX__
  constexpr int BLOCK = 8;
  // the sliding window is 4x5 and can obtain 2x3 results！ for AVX
  constexpr int window_h = 4;
  constexpr int window_w = 5;

#else
  constexpr int BLOCK = 4;
  constexpr int window_h = 4;
  constexpr int window_w = 5;
#endif

  // The maximum value of the upper left corner of the
  // sliding window in h dimension
  int new_ih;
  int new_iw;
  int new_ih_start;
  int new_iw_start;
  if (ph == 0 && pw == 0) {
    // 2 is the stride_h of sliding window
    // 3 is the stride_w of sliding window
    new_ih = (ih - window_h) / 2 * 2;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
  } else if (ph == 1 && pw == 1) {
    new_ih = (ih - window_h) / 2 * 2;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s1 with padding = 0 or 1";
  }

push(rax);
push(rcx);
push(rbx);
push(rdx);
push(r8);
push(r9);
push(r10);
push(r11);
push(r12);
push(r13);
push(r14);
push(rdi);
push(r15);


  Xbyak::Label ih_loop;
  Xbyak::Label iw_loop;
  using reg64_t = const Xbyak::Reg64;
  reg64_t ih_iter  = rax; reg64_t iw_iter  = rcx;
  

  reg64_t input_address  = rdx; mov(input_address , ptr[rdi + 8 * 0]);
  reg64_t kernel_address = rbx; mov(kernel_address , ptr[rdi + 8 * 1]);
  reg64_t output_address0 = r8; 
  reg64_t output_address1 = r9;
  mov(output_address0, ptr[rdi + 8 * 2]);
  mov(output_address1, output_address0);
  add(output_address1, ow * BLOCK * sizeof(float));


  reg64_t iv0 = r10;
  reg64_t iv1 = r11;
  reg64_t iv2 = r12;
  reg64_t iv3 = r13;

  reg64_t output_address00 = r14; mov(output_address00, output_address0);
  //reg64_t output_address01; // 分别差BLOCK * sizeof(float) byte!
  //reg64_t output_address02;
  
  reg64_t output_address10 = rdi; mov(output_address10, output_address1);
  //reg64_t output_address11; // 分别差BLOCK * sizeof(float) byte!
  //reg64_t output_address12;
  reg64_t r_temp(15);

  mov(ih_iter, new_ih_start);
  L(ih_loop);
  {
    mov(iv0, input_address);
    imul(r_temp, ih_iter, iw * sizeof(float));
    add(iv0, r_temp);
    add(iv0, new_iw_start * sizeof(float));

    mov(iv1, iv0);
    mov(iv2, iv0);
    mov(iv3, iv0);
    add(iv1, 1 * iw * sizeof(float));
    add(iv2, 2 * iw * sizeof(float));
    add(iv3, 3 * iw * sizeof(float));
    mov(output_address00, output_address0);
    mov(output_address10, output_address1);


    mov(iw_iter, new_iw_start); // initialization iw_iter！
    L(iw_loop);
    {
      // Sliding windows can produce 2x3 results, I now create them
      Xbyak::Ymm res00 = ymm0; 
      vmovups(res00, ptr[output_address00 + 0 * BLOCK * sizeof(float)]);
      Xbyak::Ymm res01 = ymm1; 
      vmovups(res01, ptr[output_address00 + 1 * BLOCK * sizeof(float)]);
      Xbyak::Ymm res02 = ymm2; 
      vmovups(res02, ptr[output_address00 + 2 * BLOCK * sizeof(float)]);
      Xbyak::Ymm res10 = ymm3; 
      vmovups(res10, ptr[output_address10 + 0 * BLOCK * sizeof(float)]);
      Xbyak::Ymm res11 = ymm4; 
      vmovups(res11, ptr[output_address10 + 1 * BLOCK * sizeof(float)]);
      Xbyak::Ymm res12 = ymm5; 
      vmovups(res12, ptr[output_address10 + 2 * BLOCK * sizeof(float)]);

      Xbyak::Ymm input00 = ymm6; 
      vbroadcastss(input00, ptr[iv0 + 0 * sizeof(float)]);
      Xbyak::Ymm input02 = ymm7; 
      vbroadcastss(input02, ptr[iv0 + 1 * sizeof(float)]);
      Xbyak::Ymm input04 = ymm8; 
      vbroadcastss(input04, ptr[iv0 + 2 * sizeof(float)]);
      Xbyak::Ymm input10 = ymm9; 
      vbroadcastss(input10, ptr[iv3 + 0 * sizeof(float)]);
      Xbyak::Ymm input12 = ymm10; 
      vbroadcastss(input12, ptr[iv3 + 1 * sizeof(float)]);
      Xbyak::Ymm input14 = ymm11; 
      vbroadcastss(input14, ptr[iv3 + 2 * sizeof(float)]);
      Xbyak::Ymm w00 = ymm12; // 0行均用ymm12 
      vmovups(w00, ptr[kernel_address + 0 * BLOCK * sizeof(float)]);
      Xbyak::Ymm w20 = ymm14;  // 2行均用ymm14
      vmovups(w20, ptr[kernel_address + 6 * BLOCK * sizeof(float)]);
      vfmadd231ps(res00, w00, input00);
      vfmadd231ps(res01, w00, input02);
      vfmadd231ps(res02, w00, input04);
      vfmadd231ps(res10, w20, input10);
      vfmadd231ps(res11, w20, input12);
      vfmadd231ps(res12, w20, input14);
      vbroadcastss(input00, ptr[iv0 + 3 * sizeof(float)]);
      vbroadcastss(input10, ptr[iv3 + 3 * sizeof(float)]);
      Xbyak::Ymm w01 = ymm12; vmovups(w01, ptr[kernel_address + 1 * BLOCK * sizeof(float)]);
      Xbyak::Ymm w21 = ymm14; vmovups(w21, ptr[kernel_address + 7 * BLOCK * sizeof(float)]);
      vfmadd231ps(res00, w01, input02);
      vfmadd231ps(res01, w01, input04);
      vfmadd231ps(res02, w01, input00);
      vfmadd231ps(res10, w21, input12);
      vfmadd231ps(res11, w21, input14);
      vfmadd231ps(res12, w21, input10);
      vbroadcastss(input02, ptr[iv0 + 4 * sizeof(float)]);
      vbroadcastss(input12, ptr[iv3 + 4 * sizeof(float)]);
      Xbyak::Ymm w02 = ymm12; vmovups(w02, ptr[kernel_address + 2 * BLOCK * sizeof(float)]);
      Xbyak::Ymm w22 = ymm14; vmovups(w22, ptr[kernel_address + 8 * BLOCK * sizeof(float)]);
      vfmadd231ps(res00, w02, input04);
      vfmadd231ps(res01, w02, input00);
      vfmadd231ps(res02, w02, input02);
      vfmadd231ps(res10, w22, input14);
      vfmadd231ps(res11, w22, input10);
      vfmadd231ps(res12, w22, input12);

            // iv1: 0 1 2 3 4
            // 0,1,2 is Responsible for res00,res10
            // 1,2,3 is Responsible for res01,res11
            // 2,3,4 is Responsible for res02,res12
            vbroadcastss(input00, ptr[iv1 + 0 * sizeof(float)]);
            vbroadcastss(input02, ptr[iv1 + 1 * sizeof(float)]);
            vbroadcastss(input04, ptr[iv1 + 2 * sizeof(float)]);
            Xbyak::Ymm w10 = ymm13; 
            vmovups(w10, ptr[kernel_address + 3 * BLOCK * sizeof(float)]);
            vmovups(w00, ptr[kernel_address + 0 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w10, input00);
            vfmadd231ps(res01, w10, input02);
            vfmadd231ps(res02, w10, input04);
            vfmadd231ps(res10, w00, input00);
            vfmadd231ps(res11, w00, input02);
            vfmadd231ps(res12, w00, input04);
            vbroadcastss(input00, ptr[iv1 + 3 * sizeof(float)]);
            Xbyak::Ymm w11 = ymm13; 
            vmovups(w11, ptr[kernel_address + 4 * BLOCK * sizeof(float)]);
            vmovups(w01, ptr[kernel_address + 1 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w11, input02);
            vfmadd231ps(res01, w11, input04);
            vfmadd231ps(res02, w11, input00);
            vfmadd231ps(res10, w01, input02);
            vfmadd231ps(res11, w01, input04);
            vfmadd231ps(res12, w01, input00);
            vbroadcastss(input02, ptr[iv1 + 4 * sizeof(float)]);
            Xbyak::Ymm w12 = ymm13; 
            vmovups(w12, ptr[kernel_address + 5 * BLOCK * sizeof(float)]);
            vmovups(w02, ptr[kernel_address + 2 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w12, input04);
            vfmadd231ps(res01, w12, input00);
            vfmadd231ps(res02, w12, input02);
            vfmadd231ps(res10, w02, input04);
            vfmadd231ps(res11, w02, input00);
            vfmadd231ps(res12, w02, input02);

            // // iv2: 0 1 2 3 4
            // // 0,1,2 is Responsible for res00,res10
            // // 1,2,3 is Responsible for res01,res11
            // // 2,3,4 is Responsible for res02,res12
            vbroadcastss(input00, ptr[iv2 + 0 * sizeof(float)]);
            vbroadcastss(input02, ptr[iv2 + 1 * sizeof(float)]);
            vbroadcastss(input04, ptr[iv2 + 2 * sizeof(float)]);
            vmovups(w20, ptr[kernel_address + 6 * BLOCK * sizeof(float)]);
            vmovups(w10, ptr[kernel_address + 3 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w20, input00);
            vfmadd231ps(res01, w20, input02);
            vfmadd231ps(res02, w20, input04);
            vfmadd231ps(res10, w10, input00);
            vfmadd231ps(res11, w10, input02);
            vfmadd231ps(res12, w10, input04);
            vbroadcastss(input00, ptr[iv2 + 3 * sizeof(float)]);
            vmovups(w21, ptr[kernel_address + 7 * BLOCK * sizeof(float)]);
            vmovups(w11, ptr[kernel_address + 4 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w21, input02);
            vfmadd231ps(res01, w21, input04);
            vfmadd231ps(res02, w21, input00);
            vfmadd231ps(res10, w11, input02);
            vfmadd231ps(res11, w11, input04);
            vfmadd231ps(res12, w11, input00);
            vbroadcastss(input02, ptr[iv2 + 4 * sizeof(float)]);
            vmovups(w22, ptr[kernel_address + 8 * BLOCK * sizeof(float)]);
            vmovups(w12, ptr[kernel_address + 5 * BLOCK * sizeof(float)]);
            vfmadd231ps(res00, w22, input04);
            vfmadd231ps(res01, w22, input00);
            vfmadd231ps(res02, w22, input02);
            vfmadd231ps(res10, w12, input04);
            vfmadd231ps(res11, w12, input00);
            vfmadd231ps(res12, w12, input02);

            // Store them back
            vmovups(ptr[output_address00 + 0 * BLOCK * sizeof(float)], res00);
            vmovups(ptr[output_address00 + 1 * BLOCK * sizeof(float)], res01);
            vmovups(ptr[output_address00 + 2 * BLOCK * sizeof(float)], res02);
            vmovups(ptr[output_address10 + 0 * BLOCK * sizeof(float)], res10);
            vmovups(ptr[output_address10 + 1 * BLOCK * sizeof(float)], res11);
            vmovups(ptr[output_address10 + 2 * BLOCK * sizeof(float)], res12);

      // update some value ;
      add(iv0, 3 * sizeof(float));
      add(iv1, 3 * sizeof(float));
      add(iv2, 3 * sizeof(float));
      add(iv3, 3 * sizeof(float));
      add(output_address00, 3 * BLOCK * sizeof(float));
      add(output_address10, 3 * BLOCK * sizeof(float));
      add(iw_iter, 3);
      cmp(iw_iter, new_iw);
      jle(iw_loop);  
    }

    // update some value;
    add(ih_iter, 2);
    add(output_address0, 2 * ow * BLOCK * sizeof(float));
    add(output_address1, 2 * ow * BLOCK * sizeof(float));
    cmp(ih_iter, new_ih);
    jle(ih_loop);
  }

pop(r15);
pop(rdi);
pop(r14);
pop(r13);
pop(r12);
pop(r11);
pop(r10);
pop(r9);
pop(r8);
pop(rdx);
pop(rbx);
pop(rcx);
pop(rax);
ret();
}





void conv_direct_3x3s1::run(const float* i_data,
                       const float* trans_weight,
                       int bs,
                       int ic,
                       int ih,
                       int iw,
                       int oc,
                       int oc_expand,
                       float* o_data,
                       int oh,
                       int ow,
                       int ph,
                       int pw,
                       const float* bias,
                       lite_api::ActivationType active_type) {
  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 1;
  constexpr int stridew = 1;

#ifdef __AVX__
  constexpr int BLOCK = 8;
  // the sliding window is 4x5 and can obtain 2x3 results！ for AVX
  constexpr int window_h = 4;
  constexpr int window_w = 5;

#else
  constexpr int BLOCK = 4;
  constexpr int window_h = 4;
  constexpr int window_w = 5;
#endif

  // The maximum value of the upper left corner of the
  // sliding window in h dimension
  int new_ih;
  int new_iw;
  int new_ih_start;
  int new_iw_start;
  if (ph == 0 && pw == 0) {
    // 2 is the stride_h of sliding window
    // 3 is the stride_w of sliding window
    new_ih = (ih - window_h) / 2 * 2;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
  } else if (ph == 1 && pw == 1) {
    new_ih = (ih - window_h) / 2 * 2;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s1 with padding = 0 or 1";
  }

  // [0,o_left) in output map needs Special treatment (Left boundary)
  // [o_right, ow) in output map needs Special treatment (Right boundary)
  // [0,o_upper) same as above (Upper boundary)
  // [o_down, oh) same as above (Lower boundary)
  int o_left = (new_iw_start + pw) / 1;
  int o_right = (new_iw + pw) / 1 + 3;
  int o_upper = (new_ih_start + ph) / 1;
  int o_down = (new_ih + ph) / 1 + 2;

  // The number of channels of convolution kernel
  // and the number of input channels are always the same !
  int wc = ic;

  int ichw = ic * ih * iw;
  int ihw = ih * iw;
  int wchw = wc * wh * ww;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;
  int ochw = oc * oh * ow;
  int owB = ow * BLOCK;
  int trans_out_size = oc_expand * ohw;

  // holds the intermediate  HWC output result
  float* trans_out = static_cast<float*>(
      TargetMalloc(TARGET(kX86), sizeof(float) * trans_out_size));

  // fetch bs_i th input feature map
  for (int bs_i = 0; bs_i < bs; bs_i++) {
    memset(trans_out, 0, sizeof(float) * trans_out_size);

    // Handle upper boundary！
    // We dealt with the boundary from the beginning
    for (int oh_i = 0; oh_i < o_upper; oh_i++) {
      for (int ow_i = 0; ow_i < ow; ow_i++) {
        // oh_i and ow_i is the index of the output.
        // Next, calculate the index of their corresponding input.
        // These two are in the upper left corner of the corresponding
        // input!
        int ih_i = oh_i * strideh - ph;
        int iw_i = ow_i * stridew - pw;

        // fetch the ic_i th channel in this input feature map
        for (int ic_i = 0; ic_i < wc; ic_i++) {
          const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;

          // fetch oc_gi th group kernel,there are BLOCK kernels
          // in it. we only need to deal with its ic_i channel !
          // oc_gi is oc_group_i !
          for (int oc_gi = 0; oc_gi < oc_expand; oc_gi += BLOCK) {
            // Now, we need compute the conv of one planar feature map and BLOCK
            // planar kernel
            // the  planar feature map's starting address
            const float* kernel_start_address =
                trans_weight + oc_gi * wchw +
                ic_i * whwB;  // the first kernel's address in this BLOCK
            float* output_address =
                trans_out + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

// Let's start the convolution of 3x3!
#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
      }
    }

    // Handle lower boundary！
    for (int oh_i = o_down; oh_i < oh; oh_i++) {
      for (int ow_i = 0; ow_i < ow; ow_i++) {
        int ih_i = oh_i * strideh - ph;
        int iw_i = ow_i * stridew - pw;

        // fetch the ic_i th channel in this input feature map
        for (int ic_i = 0; ic_i < wc; ic_i++) {
          const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;
          // fetch oc_gi th group kernel,there are BLOCK kernels
          // in it. we only need to deal with its ic_i channel !
          // oc_gi is oc_group_i !
          for (int oc_gi = 0; oc_gi < oc_expand; oc_gi += BLOCK) {
            // Now, we need compute the conv of one planar feature map and BLOCK
            // planar kernel
            // the  planar feature map's starting address
            const float* kernel_start_address =
                trans_weight + oc_gi * wchw +
                ic_i * whwB;  // the first kernel's address in this BLOCK
            float* output_address =
                trans_out + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
      }
    }

    // Handle left boundary！
    for (int oh_i = 0; oh_i < oh; oh_i++) {
      if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
        continue;
      for (int ow_i = 0; ow_i < o_left; ow_i++) {
        int ih_i = oh_i * strideh - ph;
        int iw_i = ow_i * stridew - pw;

        // fetch the ic_i th channel in this input feature map
        for (int ic_i = 0; ic_i < wc; ic_i++) {
          const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;
          // fetch oc_gi th group kernel,there are BLOCK kernels
          // in it. we only need to deal with its ic_i channel !
          // oc_gi is oc_group_i !
          for (int oc_gi = 0; oc_gi < oc_expand; oc_gi += BLOCK) {
            // Now, we need compute the conv of one planar feature map and BLOCK
            // planar kernel
            // the  planar feature map's starting address
            const float* kernel_start_address =
                trans_weight + oc_gi * wchw +
                ic_i * whwB;  // the first kernel's address in this BLOCK
            float* output_address =
                trans_out + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
      }
    }
    
    // Handle right boundary！
    for (int oh_i = 0; oh_i < oh; oh_i++) {
      if ((oh_i >= 0 && oh_i < o_upper) || (oh_i >= o_down && oh_i < oh))
        continue;
      for (int ow_i = o_right; ow_i < ow; ow_i++) {
        int ih_i = oh_i * strideh - ph;
        int iw_i = ow_i * stridew - pw;

        // fetch the ic_i th channel in this input feature map
        for (int ic_i = 0; ic_i < wc; ic_i++) {
          const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;
          // fetch oc_gi th group kernel,there are BLOCK kernels
          // in it. we only need to deal with its ic_i channel !
          // oc_gi is oc_group_i !
          for (int oc_gi = 0; oc_gi < oc_expand; oc_gi += BLOCK) {
            // Now, we need compute the conv of one planar feature map and BLOCK
            // planar kernel
            // the  planar feature map's starting address
            const float* kernel_start_address =
                trans_weight + oc_gi * wchw +
                ic_i * whwB;  // the first kernel's address in this BLOCK
            float* output_address =
                trans_out + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__
            __m256 res = _mm256_loadu_ps(output_address);
#else
            __m128 res = _mm_loadu_ps(output_address);
#endif
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                int new_ih_i = ih_i + i;
                int new_iw_i = iw_i + j;
                if (new_ih_i < 0 || new_ih_i >= ih || new_iw_i < 0 ||
                    new_iw_i >= iw)
                  continue;
                const float* input_address =
                    input_start_address + new_ih_i * iw + new_iw_i;
#ifdef __AVX__
                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w =
                    _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
      }
    }

    // fetch the ic_i th channel in this input feature map
    for (int ic_i = 0; ic_i < wc; ic_i++) {
      const float* input_start_address = i_data + bs_i * ichw + ic_i * ihw;

      // fetch oc_gi th group kernel,there are BLOCK kernels
      // in it. we only need to deal with its ic_i channel !
      // oc_gi is oc_group_i !
      for (int oc_gi = 0; oc_gi < oc_expand; oc_gi += BLOCK) {
        // Now, we need compute the conv of one planar feature map and BLOCK
        // planar kernel
        // the  planar feature map's starting address
        const float* kernel_start_address =
            trans_weight + oc_gi * wchw +
            ic_i * whwB;  // the first kernel's address in this BLOCK
        float* output_start_address = trans_out + oc_gi * ohw;

/* So far, we have dealt with the special boundary,and now we begin to deal with
 * the general situation */

// prefetch the 3x3 conv kernel outside the below two Nested loop !
#ifdef __AVX__

        // Take out 9 weight values to the register
        // __m256 w00 = _mm256_loadu_ps(kernel_start_address + 0 * BLOCK);
        // __m256 w01 = _mm256_loadu_ps(kernel_start_address + 1 * BLOCK);
        // __m256 w02 = _mm256_loadu_ps(kernel_start_address + 2 * BLOCK);
        // __m256 w10 = _mm256_loadu_ps(kernel_start_address + 3 * BLOCK);
        // __m256 w11 = _mm256_loadu_ps(kernel_start_address + 4 * BLOCK);
        // __m256 w12 = _mm256_loadu_ps(kernel_start_address + 5 * BLOCK);
        // __m256 w20 = _mm256_loadu_ps(kernel_start_address + 6 * BLOCK);
        // __m256 w21 = _mm256_loadu_ps(kernel_start_address + 7 * BLOCK);
        // __m256 w22 = _mm256_loadu_ps(kernel_start_address + 8 * BLOCK);
#else
        // SSE version
#endif

        // one sliding window cangenerate 2x3 results
        // below is the two line's first address the first window generated!
        float* output_address0 = output_start_address +
                                 (new_ih_start + ph) / strideh * ow * BLOCK +
                                 (new_iw_start + pw) / stridew * BLOCK;
        //float* output_address1 = output_address0 + ow * BLOCK;


        jit_3x3s1_param param;
        param.input_address = input_start_address;
        param.kernel_address = kernel_start_address;
        param.output_address = output_address0;
        void (*f)(jit_3x3s1_param*) = getCode<void (*)(jit_3x3s1_param*)>();
        f(&param);
        // 下面需要删掉

      }
    }

    // we always assume oc % BLOCK == 0!
    // convert trans_out(HWC) to o_data(CHW)!
    for (int oc_gi = 0; oc_gi < oc; oc_gi += BLOCK) {
      for (int oh_i = 0; oh_i < oh; oh_i++) {
        for (int ow_i = 0; ow_i < ow / BLOCK * BLOCK; ow_i += BLOCK) {
          // trans_out's start_index, we need fetch 8x8 element;
          float* from_address =
              trans_out + oc_gi * ohw + oh_i * owB + ow_i * BLOCK;

#ifdef __AVX__
          __m256 row0 = _mm256_loadu_ps(from_address + 0 * BLOCK);
          __m256 row1 = _mm256_loadu_ps(from_address + 1 * BLOCK);
          __m256 row2 = _mm256_loadu_ps(from_address + 2 * BLOCK);
          __m256 row3 = _mm256_loadu_ps(from_address + 3 * BLOCK);
          __m256 row4 = _mm256_loadu_ps(from_address + 4 * BLOCK);
          __m256 row5 = _mm256_loadu_ps(from_address + 5 * BLOCK);
          __m256 row6 = _mm256_loadu_ps(from_address + 6 * BLOCK);
          __m256 row7 = _mm256_loadu_ps(from_address + 7 * BLOCK);
          transpose8_ps(row0, row1, row2, row3, row4, row5, row6, row7);
#else

          __m128 row0 = _mm_loadu_ps(from_address + 0 * BLOCK);
          __m128 row1 = _mm_loadu_ps(from_address + 1 * BLOCK);
          __m128 row2 = _mm_loadu_ps(from_address + 2 * BLOCK);
          __m128 row3 = _mm_loadu_ps(from_address + 3 * BLOCK);
          _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

#endif

          if (bias != nullptr) {
#ifdef __AVX__
            row0 = _mm256_add_ps(row0, _mm256_set1_ps(bias[oc_gi + 0]));
            row1 = _mm256_add_ps(row1, _mm256_set1_ps(bias[oc_gi + 1]));
            row2 = _mm256_add_ps(row2, _mm256_set1_ps(bias[oc_gi + 2]));
            row3 = _mm256_add_ps(row3, _mm256_set1_ps(bias[oc_gi + 3]));
            row4 = _mm256_add_ps(row4, _mm256_set1_ps(bias[oc_gi + 4]));
            row5 = _mm256_add_ps(row5, _mm256_set1_ps(bias[oc_gi + 5]));
            row6 = _mm256_add_ps(row6, _mm256_set1_ps(bias[oc_gi + 6]));
            row7 = _mm256_add_ps(row7, _mm256_set1_ps(bias[oc_gi + 7]));
#else
            row0 = _mm_add_ps(row0, _mm_set1_ps(bias[oc_gi + 0]));
            row1 = _mm_add_ps(row1, _mm_set1_ps(bias[oc_gi + 1]));
            row2 = _mm_add_ps(row2, _mm_set1_ps(bias[oc_gi + 2]));
            row3 = _mm_add_ps(row3, _mm_set1_ps(bias[oc_gi + 3]));
#endif
          }

          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            row0 = _mm256_max_ps(row0, vzero);
            row1 = _mm256_max_ps(row1, vzero);
            row2 = _mm256_max_ps(row2, vzero);
            row3 = _mm256_max_ps(row3, vzero);
            row4 = _mm256_max_ps(row4, vzero);
            row5 = _mm256_max_ps(row5, vzero);
            row6 = _mm256_max_ps(row6, vzero);
            row7 = _mm256_max_ps(row7, vzero);
#else
            row0 = _mm_max_ps(row0, _mm_set1_ps(0.f));
            row1 = _mm_max_ps(row1, _mm_set1_ps(0.f));
            row2 = _mm_max_ps(row2, _mm_set1_ps(0.f));
            row3 = _mm_max_ps(row3, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            __m256 vzero = _mm256_set1_ps(0.f);
            __m256 vsix = _mm256_set1_ps(6.f);
            row0 = _mm256_max_ps(row0, vzero);
            row1 = _mm256_max_ps(row1, vzero);
            row2 = _mm256_max_ps(row2, vzero);
            row3 = _mm256_max_ps(row3, vzero);
            row4 = _mm256_max_ps(row4, vzero);
            row5 = _mm256_max_ps(row5, vzero);
            row6 = _mm256_max_ps(row6, vzero);
            row7 = _mm256_max_ps(row7, vzero);
            row0 = _mm256_min_ps(row0, vsix);
            row1 = _mm256_min_ps(row1, vsix);
            row2 = _mm256_min_ps(row2, vsix);
            row3 = _mm256_min_ps(row3, vsix);
            row4 = _mm256_min_ps(row4, vsix);
            row5 = _mm256_min_ps(row5, vsix);
            row6 = _mm256_min_ps(row6, vsix);
            row7 = _mm256_min_ps(row7, vsix);

#else
            __m128 vzero = _mm_set1_ps(0.f);
            __m128 vsix = _mm_set1_ps(6.f);
            row0 = _mm_max_ps(row0, vzero);
            row1 = _mm_max_ps(row1, vzero);
            row2 = _mm_max_ps(row2, vzero);
            row3 = _mm_max_ps(row3, vzero);
            row0 = _mm_min_ps(row0, vsix);
            row1 = _mm_min_ps(row1, vsix);
            row2 = _mm_min_ps(row2, vsix);
            row3 = _mm_min_ps(row3, vsix);
#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }

          float* dst_address =
              o_data + bs_i * ochw + oc_gi * ohw + oh_i * ow + ow_i;
#ifdef __AVX__
          _mm256_storeu_ps(dst_address + 0 * ohw, row0);
          _mm256_storeu_ps(dst_address + 1 * ohw, row1);
          _mm256_storeu_ps(dst_address + 2 * ohw, row2);
          _mm256_storeu_ps(dst_address + 3 * ohw, row3);
          _mm256_storeu_ps(dst_address + 4 * ohw, row4);
          _mm256_storeu_ps(dst_address + 5 * ohw, row5);
          _mm256_storeu_ps(dst_address + 6 * ohw, row6);
          _mm256_storeu_ps(dst_address + 7 * ohw, row7);
#else
          _mm_storeu_ps(dst_address + 0 * ohw, row0);
          _mm_storeu_ps(dst_address + 1 * ohw, row1);
          _mm_storeu_ps(dst_address + 2 * ohw, row2);
          _mm_storeu_ps(dst_address + 3 * ohw, row3);
#endif
        }

        for (int ow_i = ow / BLOCK * BLOCK; ow_i < ow; ow_i++) {
          // trans_out
          float* from_address =
              trans_out + oc_gi * ohw + oh_i * owB + ow_i * BLOCK;
          float* dst_address =
              o_data + bs_i * ochw + oc_gi * ohw + oh_i * ow + ow_i;
#ifdef __AVX__
          __m256 row = _mm256_loadu_ps(from_address);
#else
          __m128 row = _mm_loadu_ps(from_address);
#endif
          if (bias != nullptr) {
#ifdef __AVX__
            row = _mm256_add_ps(row, _mm256_loadu_ps(&bias[oc_gi]));
#else
            row = _mm_add_ps(row, _mm_loadu_ps(&bias[oc_gi]));
#endif
          }
          if (active_type == lite_api::ActivationType::kRelu) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
#endif
          } else if (active_type == lite_api::ActivationType::kRelu6) {
#ifdef __AVX__
            row = _mm256_max_ps(row, _mm256_set1_ps(0.f));
            row = _mm256_min_ps(row, _mm256_set1_ps(6.f));
#else
            row = _mm_max_ps(row, _mm_set1_ps(0.f));
            row = _mm_min_ps(row, _mm_set1_ps(6.f));
#endif
          } else if (active_type == lite_api::ActivationType::kIndentity) {
          } else {
            LOG(FATAL) << "[X86] unsupported Activation type";
          }
#ifdef __AVX__
          *(dst_address + 0 * oh * ow) = (reinterpret_cast<float*>(&row))[0];
          *(dst_address + 1 * oh * ow) = (reinterpret_cast<float*>(&row))[1];
          *(dst_address + 2 * oh * ow) = (reinterpret_cast<float*>(&row))[2];
          *(dst_address + 3 * oh * ow) = (reinterpret_cast<float*>(&row))[3];
          *(dst_address + 4 * oh * ow) = (reinterpret_cast<float*>(&row))[4];
          *(dst_address + 5 * oh * ow) = (reinterpret_cast<float*>(&row))[5];
          *(dst_address + 6 * oh * ow) = (reinterpret_cast<float*>(&row))[6];
          *(dst_address + 7 * oh * ow) = (reinterpret_cast<float*>(&row))[7];
#else
          *(dst_address + 0 * oh * ow) = (reinterpret_cast<float*>(&row))[0];
          *(dst_address + 1 * oh * ow) = (reinterpret_cast<float*>(&row))[1];
          *(dst_address + 2 * oh * ow) = (reinterpret_cast<float*>(&row))[2];
          *(dst_address + 3 * oh * ow) = (reinterpret_cast<float*>(&row))[3];
#endif
        }
      }
    }
  }
  	
  TargetFree(TARGET(kX86), trans_out);
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
