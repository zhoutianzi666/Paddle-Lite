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
#include "lite/backends/x86/math/conv_direct.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "lite/backends/x86/math/conv_utils.h"
#include "lite/core/context.h"
#ifdef __AVX__
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

#include "lite/backends/x86/jit/gen/jitcode.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

struct  jit_param
{
  const float* input_row_address;
  const float* kernel_row_address;
  float* output_row_address;
};

conv_direct_3x3s2Code ::conv_direct_3x3s2Code
                      (int ic,
                       int ih,
                       int iw,
                       int oc,
                       int oc_expand,
                       int oh,
                       int ow,
                       int ph,
                       int pw): Xbyak::CodeGenerator(8192, Xbyak::AutoGrow)
{

  constexpr int ww = 3;
  constexpr int wh = 3;
  //constexpr int strideh = 2;
  //constexpr int stridew = 2;

#ifdef __AVX__
  constexpr int BLOCK = 8;
  // the sliding window is 3x7 and can obtain 1x3 results！ for AVX
  constexpr int window_h = 3;
  constexpr int window_w = 5;

#else
  constexpr int BLOCK = 4;
  constexpr int window_h = 3;
  constexpr int window_w = 5;
#endif

  // The maximum value of the upper left corner of the
  // sliding window in h dimension
  int new_ih;
  int new_iw;
  int new_ih_start;
  int new_iw_start;
  if (ph == 0 && pw == 0) {
    // 4 is the stride_h of sliding window
    // 6 is the stride_w of sliding window
    new_ih = (ih - window_h) / 1 * 1;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
  } else if (ph == 1 && pw == 1) {
    new_ih = (ih - window_h) / 1 * 1;
    new_iw = (iw - window_w) / 3 * 3;

    new_ih_start = 0;
    new_iw_start = 0;
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s2 with padding = 0 or 1";
  }

  // // The number of channels of convolution kernel
  // // and the number of input channels are always the same !
  int wc = ic;

  int ihw = ih * iw;
  int wchw = wc * wh * ww;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;
 
  using reg64_t = const Xbyak::Reg64;

  push(rdx);
  push(rcx);
  push(r8);
  push(r9);
  push(r10);
  push(r11);

  reg64_t iw_iter  = rcx;
  reg64_t oc_iter  = rdx;

  reg64_t input_row_address_xb  = r8; mov(input_row_address_xb , ptr[rdi + 8 * 0]);
  reg64_t kernel_address_xb = r9;     mov(kernel_address_xb , ptr[rdi + 8 * 1]);
  reg64_t output_row_address_xb = r10;mov(output_row_address_xb, ptr[rdi + 8 * 2]);
  // 上面三个是非常重要的东西
  // 其中input_start_address是从new_ih_start开始的哦!
  // output_start_address_xb肯定不是从0开始的哈！

reg64_t reg_tmp = r11;

 Xbyak::Label oc_loop; 
 int oc_loop_n = oc_expand / 32; // every 32 output channels are a loop
 int oc_remain = oc_expand % 32;
 xor_(oc_iter, oc_iter);
 int temp;

// compute 这个函数的参数oc_group的意思是我这次要处理多少个卷积核
// 
auto compute = [=,&temp](int oc_group) {
  
  for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
      Xbyak::Ymm res0(oc_gi / BLOCK * 3 + 0);
      Xbyak::Ymm res1(oc_gi / BLOCK * 3 + 1);
      Xbyak::Ymm res2(oc_gi / BLOCK * 3 + 2);
      vxorps(res0,res0,res0);
      vxorps(res1,res1,res1);
      vxorps(res2,res2,res2);
    }

// 这里开始处理这6个输入数字和卷积核心的卷积！
for (int wh_i = 0; wh_i < wh; wh_i ++){// oneDNN 是不展开的！, 这里先展开一下吧！
  for (int ww_i = 0; ww_i < ww; ww_i ++){// 卷积核有三列,但是我每次只拿一颗！
    for (int ic_i = 0; ic_i < wc; ic_i++) {// inchannel哦！

    // get three input data
    Xbyak::Ymm input00 = ymm12;
    Xbyak::Ymm input02 = ymm13;
    Xbyak::Ymm input04 = ymm14;
    temp = (ww_i + 0 + wh_i * iw + ic_i * ihw) * sizeof(float);
    vbroadcastss(input00, ptr[input_row_address_xb + temp]);
    temp = (ww_i + 1 + wh_i * iw + ic_i * ihw) * sizeof(float);
    vbroadcastss(input02, ptr[input_row_address_xb + temp]);
    temp = (ww_i + 2 + wh_i * iw + ic_i * ihw) * sizeof(float);
    vbroadcastss(input04, ptr[input_row_address_xb + temp]);

     for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {

      //  接着搞一个卷积核
      Xbyak::Ymm kernel = ymm15;
      temp = (oc_gi * wchw + ic_i * whwB + ww_i * BLOCK + wh_i * ww * BLOCK) * sizeof(float);
      vmovups(kernel, ptr[kernel_address_xb + temp]);

      // 接着获得3个输出结果
      // 拿着ymm15和3个输入搞起来！
      Xbyak::Ymm res0(oc_gi / BLOCK * 3 + 0);
      Xbyak::Ymm res1(oc_gi / BLOCK * 3 + 1);
      Xbyak::Ymm res2(oc_gi / BLOCK * 3 + 2);
      vfmadd231ps(res0, kernel, input00);
      vfmadd231ps(res1, kernel, input02);
      vfmadd231ps(res2, kernel, input04);
    }
  }
 }
}

    // 这里需要store输出了哦！12个输出!
for (int oc_gi = 0; oc_gi < oc_group; oc_gi += BLOCK) {
      Xbyak::Ymm res0(oc_gi / BLOCK * 3 + 0);
      Xbyak::Ymm res1(oc_gi / BLOCK * 3 + 1);
      Xbyak::Ymm res2(oc_gi / BLOCK * 3 + 2);
      
      temp = (oc_gi * ohw + 0 * BLOCK) * sizeof(float);
      vmovups(ptr[output_row_address_xb + temp], res0);
      temp = (oc_gi * ohw + 1 * BLOCK) * sizeof(float);
      vmovups(ptr[output_row_address_xb + temp], res1);
      temp = (oc_gi * ohw + 2 * BLOCK) * sizeof(float);
      vmovups(ptr[output_row_address_xb + temp], res2);
    }
};

if (oc_loop_n >= 1)
{
L(oc_loop);
{
  xor_(iw_iter, iw_iter);
  add(iw_iter, new_iw_start);
  Xbyak::Label iw_loop;




  L(iw_loop);
  {
    compute(32);

    add(input_row_address_xb, 3 * sizeof(float));
    add(output_row_address_xb, 3 * BLOCK * sizeof(float));
    add(iw_iter, 3);
    cmp(iw_iter, new_iw);
    jle(iw_loop,T_NEAR);
  }

  inc(oc_iter);

  mov(input_row_address_xb , ptr[rdi + 8 * 0]);
  mov(kernel_address_xb , ptr[rdi + 8 * 1]);
  mov(output_row_address_xb, ptr[rdi + 8 * 2]);

  mov(reg_tmp,whwB * wc * sizeof(float) * 4);
  imul(reg_tmp, oc_iter);
  add(kernel_address_xb , reg_tmp);

  // update output_row_address_xb
  mov(reg_tmp, oh * ow * BLOCK * sizeof(float) * 4);
  imul(reg_tmp, oc_iter);
  add(output_row_address_xb, reg_tmp);

  cmp(oc_iter, oc_loop_n);
  jl(oc_loop);

}
}



/* deal with the remain oc*/

if( oc_remain != 0)
{
  Xbyak::Label iw2_loop;
  mov(iw_iter, new_iw_start);

  L(iw2_loop);
  {
    compute(oc_remain);

    add(input_row_address_xb, 3 * sizeof(float));
    add(output_row_address_xb, 3 * BLOCK * sizeof(float));
    add(iw_iter, 3);
    cmp(iw_iter, new_iw);
    jle(iw2_loop,T_NEAR);
  }

}


pop(r11);
pop(r10);
pop(r9);
pop(r8);
pop(rcx);
pop(rdx);
ret();

}

void conv_direct_3x3s2Code::run
                      (const float* i_data,
                      const float* trans_weight,
                      int bs,
                       int ic,
                       int ih,
                       int iw,
                       int oc,
                       int oc_expand,
                       float* o_data,
                       float* trans_out,
                       int oh,
                       int ow,
                       int ph,
                       int pw,
                       const float* bias,
                       lite_api::ActivationType active_type)
{
  constexpr int ww = 3;
  constexpr int wh = 3;
  constexpr int strideh = 1;
  constexpr int stridew = 1;

#ifdef __AVX__
  constexpr int BLOCK = 8;
  // the sliding window is 4x4 and can obtain 1x3 results！ for AVX
  constexpr int window_h = 3;
  constexpr int window_w = 5;

#else
  constexpr int BLOCK = 4;
  constexpr int window_h = 3;
  constexpr int window_w = 5;
#endif

  // The maximum value of the upper left corner of the
  // sliding window in h dimension
  int new_ih;
  int new_iw;
  int new_ih_start;
  int new_iw_start;
  if (ph == 0 && pw == 0) {
    // 4 is the stride_h of sliding window
    // 6 is the stride_w of sliding window
    new_ih = (ih - window_h) / 1 * 1;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;

  } else if (ph == 1 && pw == 1) {
    new_ih = (ih - window_h) / 1 * 1;
    new_iw = (iw - window_w) / 3 * 3;
    new_ih_start = 0;
    new_iw_start = 0;
    
  } else {
    LOG(FATAL) << "[X86] conv_direct only support 3x3s2 with padding = 0 or 1";
  }

  // [0,o_left) in output map needs Special treatment (Left boundary)
  // [o_right, ow) in output map needs Special treatment (Right boundary)
  // [0,o_upper) same as above (Upper boundary)
  // [o_down, oh) same as above (Lower boundary)
  int o_left = (new_iw_start + pw) / 1;
  int o_right = (new_iw + pw) / 1 + 3;
  int o_upper = (new_ih_start + ph) / 1;
  int o_down = (new_ih + ph) / 1 + 1;
  //std::cout << o_right << std::endl;
  // The number of channels of convolution kernel
  // and the number of input channels are always the same !
  int wc = ic;

  int ichw = ic * ih * iw;
  int ihw = ih * iw;
  int wchw = wc * wh * ww;
  int whwB = wh * ww * BLOCK;
  int ohw = oh * ow;
  int ochw = oc * oh * ow;

  // fetch bs_i th input feature map
  for (int bs_i = 0; bs_i < bs; bs_i++) {

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
                trans_out + bs_i * ochw + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

// Let's start the convolution of 3x3!
#ifdef __AVX__

// Xbyak
            // Xbyak::Ymm res_xb = ymm0;
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(res_xb, yword[pointer]);

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
                // mov(pointer ,(uint64_t)(input_address));
                // Xbyak::Ymm input_xb = ymm1;
                // vbroadcastss(input_xb, dword[pointer]);
                // mov(pointer, (uint64_t)(kernel_start_address + (i * 3 + j) * BLOCK));
                // Xbyak::Ymm w_xb = ymm2;
                // vmovups(w_xb, yword[pointer]);
                // vfmadd231ps(res_xb, input_xb, w_xb);

                __m256 input = _mm256_set1_ps(*input_address);
                __m256 w = _mm256_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm256_fmadd_ps(input, w, res);
#else
                __m128 input = _mm_set1_ps(*input_address);
                __m128 w =
                    _mm_loadu_ps(kernel_start_address + (i * 3 + j) * BLOCK);
                res = _mm_fmadd_ps(input, w, res);
#endif
              }
#ifdef __AVX__
            
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(yword[pointer], res_xb);
            
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
                trans_out + bs_i * ochw + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__

// Xbyak
            // Xbyak::Ymm res_xb = ymm0;
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(res_xb, yword[pointer]);

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

                // mov(pointer ,(uint64_t)(input_address));
                // Xbyak::Ymm input_xb = ymm1;
                // vbroadcastss(input_xb, dword[pointer]);
                // mov(pointer, (uint64_t)(kernel_start_address + (i * 3 + j) * BLOCK));
                // Xbyak::Ymm w_xb = ymm2;
                // vmovups(w_xb, yword[pointer]);
                // vfmadd231ps(res_xb, input_xb, w_xb);

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

            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(yword[pointer], res_xb);

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
                trans_out + bs_i * ochw + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__

// Xbyak
            // Xbyak::Ymm res_xb = ymm0;
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(res_xb, yword[pointer]);

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


                // mov(pointer ,(uint64_t)(input_address));
                // Xbyak::Ymm input_xb = ymm1;
                // vbroadcastss(input_xb, dword[pointer]);
                // mov(pointer, (uint64_t)(kernel_start_address + (i * 3 + j) * BLOCK));
                // Xbyak::Ymm w_xb = ymm2;
                // vmovups(w_xb, yword[pointer]);
                // vfmadd231ps(res_xb, input_xb, w_xb);


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
            
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(yword[pointer], res_xb);

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
                trans_out + bs_i * ochw + oc_gi * ohw + oh_i * ow * BLOCK + ow_i * BLOCK;

#ifdef __AVX__

// Xbyak
            // Xbyak::Ymm res_xb = ymm0;
            // mov(pointer ,(uint64_t)(output_address));
            // vmovups(res_xb, yword[pointer]);

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

                // mov(pointer ,(uint64_t)(input_address));
                // Xbyak::Ymm input_xb = ymm1;
                // vbroadcastss(input_xb, dword[pointer]);
                // mov(pointer, (uint64_t)(kernel_start_address + (i * 3 + j) * BLOCK));
                // Xbyak::Ymm w_xb = ymm2;
                // vmovups(w_xb, yword[pointer]);
                // vfmadd231ps(res_xb, input_xb, w_xb);


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
            
            //mov(pointer ,(uint64_t)(output_address));
            //vmovups(yword[pointer], res_xb);

            _mm256_storeu_ps(output_address, res);
#else
            _mm_storeu_ps(output_address, res);
#endif
          }
        }
      }
    }

/*-----------------------new way--------------------------------------*/

    const float* input_row_address = i_data + bs_i * ichw + new_iw_start + new_ih_start * iw;
    float* output_row_address = trans_out + (new_ih_start + ph) / strideh * ow * BLOCK + (new_iw_start + pw) / stridew * BLOCK;

    for (int ih_i = new_ih_start; ih_i <= new_ih; ih_i += 1, 
                                   output_row_address += ow * BLOCK,
                                   input_row_address += iw) {

        jit_param param;
        param.input_row_address = input_row_address;
        param.kernel_row_address = trans_weight;
        param.output_row_address = output_row_address;
        
        void (*f)(jit_param*) = getCode<void (*)(jit_param*)>();
        f(&param);
        
  }
}

}

void conv_direct_3x3s2_tranpose_out(int bs,
                       int oc,
                       float* o_data,
                       float* trans_out,
                       int oh,
                       int ow,
                       const float* bias,
                       lite_api::ActivationType active_type){ 

#ifdef __AVX__
  constexpr int BLOCK = 8;
#else
  constexpr int BLOCK = 4;
#endif

  int ohw = oh * ow;
  int ochw = oc * oh * ow;
  int owB = ow * BLOCK;


// we always assume oc % BLOCK == 0!
// convert trans_out(HWC) to o_data(CHW)!
// fetch bs_i th input feature map

  for (int bs_i = 0; bs_i < bs; bs_i++) {
    for (int oc_gi = 0; oc_gi < oc; oc_gi += BLOCK) {
      for (int oh_i = 0; oh_i < oh; oh_i++) {
        for (int ow_i = 0; ow_i < ow / BLOCK * BLOCK; ow_i += BLOCK) {
          // trans_out's start_index, we need fetch 8x8 element;
          float* from_address =
              trans_out + bs_i * oc * ohw + oc_gi * ohw + oh_i * owB + ow_i * BLOCK;

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
              trans_out + bs_i * ochw + oc_gi * ohw + oh_i * owB + ow_i * BLOCK;
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
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
