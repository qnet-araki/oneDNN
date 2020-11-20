/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/jit_uni_batch_normalization.hpp"

#define CG CodeGeneratorAArch64
#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

using namespace memory_tracking::names;

using namespace Xbyak_aarch64;
namespace barrier = simple_barrier;

using acc_data_t = float;

template <cpu_isa_t isa>
struct jit_bnorm_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size, spat_size_loc;
        size_t S_s, S_tail;
        size_t is_cblk_tail;
        acc_data_t chan_size, eps, one;
        const acc_data_t *scale_shift;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_scale_shift;
        const void *src, *dst;
        const void *diff_src, *diff_dst;
        const acc_data_t *rbuf1, *rbuf2;
        const uint8_t *ws;
        barrier::ctx_64_t *barrier;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    const int vlen = isa == asimd ? 32 : cpu_isa_traits<isa>::vlen;
    int vlen_spat_data_; // set by ctor depending on data type (BF16 or FP32);

    const batch_normalization_pd_t *bdesc_;
    bool is_spatial_thr_;
    bool is_nspc_;
    bool is_bf16_;

    XReg rax = x0;
    XReg rcx = x1;
    XReg rdx = x2;
    XReg rbx = x3;
    XReg rsp = sp;
    XReg rbp = x5;
    XReg rsi = x6;
    XReg rdi = x7;

    XReg reg_param = abi_param1;

    XReg reg_scale_shift = rbx;
    XReg reg_rbuf1 = x1;
    XReg reg_rbuf2 = rdx;
    XReg reg_coff_max_fwd_copy = reg_rbuf2;

    XReg reg_mean = rbp;
    XReg reg_var = reg_param;
    XReg reg_diff_scale_shift = x7;
    XReg reg_coff_max_bwd_copy = reg_diff_scale_shift;

    XReg reg_coff = x8;
    XReg reg_coff_max = x9;
    XReg reg_soff = x10;
    XReg reg_soff_max = x11;
    XReg reg_ctr = x12;
    XReg reg_roff = x13;

    XReg reg_mb_stride_Bc = x14;
    XReg reg_soff_nspc = reg_mb_stride_Bc;

    XReg reg_src = x15;
    XReg reg_diff_src = reg_rbuf1;
    XReg reg_dst = rsi;
    XReg reg_diff_dst = reg_dst;

    XReg reg_tmp_off = reg_roff;

    // Reuse loop counters
    XReg reg_bar = reg_coff;
    XReg reg_nnthr = reg_soff; // must be usable w/ loops over coff
    XReg reg_tmp = reg_ctr;

    // Relu section
    bool with_relu, with_relu_inf_only;
    ZReg vzero {0}; // Index 0 is temporal value. is_fwd() ? vdiff_beta : vbeta
    XReg reg_ws = reg_roff;
    Label l_relu_mask_avx2;
    PReg kstore_mask = PReg(1);

    // channel tail processing
    PReg ktail_mask = PReg(2);

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_512 = p7;
    PReg p_lsb_256 = p6;
    PReg p_lsb_128 = p5;
    PReg p_tmp0 = p8;

//    XReg x_translator_stack {22};
    XReg x_tmp_0 {23};
    XReg x_tmp_1 {24};
    XReg x_tmp_2 {25};

    WReg w_tmp_0 {23};
    WReg w_tmp_1 {24};
    WReg w_tmp_2 {25};
    WReg w_tmp_3 {26};

    const std::vector<XReg> x_tmp_vec = {x_tmp_0, x_tmp_1, x_tmp_2};
    constexpr static int x_tmp_vec_size = 3;

    const std::vector<WReg> w_tmp_vec = {w_tmp_0, w_tmp_1, w_tmp_2, w_tmp_3};
    constexpr static int w_tmp_vec_size = 4;

#if 0
    // FP32->BF16 emulation
    bf16_emulation_t *bf16_emu_ {nullptr};
    XReg reg_bf16_tmp = reg_tmp;
    ZReg bf16_emu_reserved_1 = z16;
    ZReg bf16_emu_reserved_2 = z17;
    ZReg bf16_emu_reserved_3 = z18;
    ZReg bf16_emu_reserved_4 = z19;
#endif // #if 0

    size_t unroll_blocks;
    size_t unroll_regs;

    ZReg vbuf = ZReg(isa == sve_512 ? 20 : 5);
    ZReg vdiff_beta = ZReg(isa == sve_512 ? 21 : 6);
    ZReg vdiff_gamma = ZReg(isa == sve_512 ? 22 : 7);
    ZReg vsqrtvar = ZReg(isa == sve_512 ? 23 : 8);
    ZReg vone = ZReg(isa == sve_512 ? 24 : 9);
    ZReg vmean = ZReg(isa == sve_512 ? 25 : 10);
    ZReg vgamma = ZReg(isa == sve_512 ? 26 : 11);
    ZReg vbeta = ZReg(isa == sve_512 ? 27 : 12);
    ZReg veps = ZReg(isa == sve_512 ? 28 : 13);
    ZReg vchan_size = ZReg(isa == sve_512 ? 29 : 14);

    const std::vector<uint32_t> tmp_vec_idx
            = {31, isa == sve_512 ? 20 : 5};

    /* Caution: Chose predicate registers not used by x64's implementation. */
    ZReg z_tmp0 = z31;
    ZReg z_tmp1 = isa == sve_512 ? z20 : z5;
    ZReg z_relu_mask_avx2 = z30;

    const std::vector<ZReg> z_tmp_vec = {z_tmp0, z_tmp1};
    constexpr static int z_tmp_vec_size = 2;

    size_t t0_pf_offt;
    size_t t1_pf_offt;
    size_t spat_size;
    size_t chan_data_offt;
    size_t spat_step;
    size_t mb_offt;
    size_t ws_mb_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_diff_scale_shift = 48,
        stack_off_ws = 56,
        stack_off_barrier = 64,
        stack_off_spat_size_loc = 72,
        stack_off_s_s = 80,
        stack_off_s_tail = 88,
        stack_off_is_cblk_tail = 96,
        stack_off_ws_off_copy = 104,
        stack_size_required = 112,
    };

    int bit_shift() { return 5 - is_bf16_; }

    bool stream_store_supported() { return !is_bf16_; }

    bool is_c_padded() const {
        const memory_desc_wrapper data_d(bdesc_->src_md());
        return bdesc_->C() != data_d.padded_dims()[1];
    }

    void compute_static_strides() {
        spat_size = bdesc_->D() * bdesc_->W() * bdesc_->H();
        chan_data_offt = bdesc_->C() * sizeof(acc_data_t);
        spat_step
                = is_nspc_ ? chan_data_offt / (1 + is_bf16_) : vlen_spat_data_;
        mb_offt = spat_step * spat_size;
        ws_mb_offt = (spat_step / (is_bf16_ ? 16 : 32)) * spat_size;

        if (false) {
            t0_pf_offt = 4096;
            t1_pf_offt = 0;
        } else {
            t0_pf_offt = 0;
            t1_pf_offt = 0;
        }
    }

    void load_common_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
#define PARAM_OFF_DIFF(x, y) \
    (static_cast<int32_t>(PARAM_OFF(x)) - static_cast<int32_t>(PARAM_OFF(y)))
#define LDR_PARAM(r, x, y) \
    assert(-256 <= PARAM_OFF_DIFF(x, y) && PARAM_OFF_DIFF(x, y) <= 255); \
    ldr(XReg(IDX(r)), pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(x, y)))
#define LDR_PARAM_TMP(x, y) \
    assert(-256 <= PARAM_OFF_DIFF(x, y) && PARAM_OFF_DIFF(x, y) <= 255); \
    ldr(x_tmp_0, pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(x, y)));
#define STR_PARAM_TMP(x, y) \
    assert(-256 <= static_cast<int32_t>(x) - static_cast<int32_t>(y) \
            && static_cast<int32_t>(x) - static_cast<int32_t>(y) <= 256); \
    str(x_tmp_0, pre_ptr(x_tmp_sp, x - y));

        mov(X_DEFAULT_ADDR, XReg(IDX(reg_param)));
        ldr(XReg(IDX(reg_rbuf1)), pre_ptr(X_DEFAULT_ADDR, PARAM_OFF(rbuf1)));
        if (bdesc_->is_bwd()) {
            LDR_PARAM(reg_rbuf2, rbuf2, rbuf1);
            LDR_PARAM(reg_coff_max, coff_max, rbuf2);
        } else {
            LDR_PARAM(reg_coff_max, coff_max, rbuf1);
        }
        LDR_PARAM(reg_soff_max, soff_max, coff_max);
        LDR_PARAM(reg_mb_stride_Bc, mb_stride_Bc, soff_max);
        lsl(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 2);

        LDR_PARAM(reg_mean, mean, mb_stride_Bc);
        LDR_PARAM(reg_scale_shift, scale_shift, mean);

        ldr(w_tmp_1,
                pre_ptr(X_DEFAULT_ADDR,
                        PARAM_OFF_DIFF(chan_size, scale_shift)));
        ldr(w_tmp_2, pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(one, chan_size)));
        ldr(w_tmp_3, pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(eps, one)));

        if (isa == sve_512) {
            dup(ZRegS(IDX(vchan_size)), w_tmp_1);
            dup(ZRegS(IDX(vone)), w_tmp_2);
            dup(ZRegS(IDX(veps)), w_tmp_3);
#if 0
            if (vchan_size.isYMM()) {
                mov(ZRegS(IDX(vchan_size)), P_MSB_256 / T_m, 0);
                mov(ZRegS(IDX(vone)), P_MSB_256 / T_m, 0);
                mov(ZRegS(IDX(veps)), P_MSB_256 / T_m, 0);
            }
#endif // #if 0
        } else {
            dup(VReg4S(IDX(vchan_size)), w_tmp_1);
            dup(VReg4S(IDX(vone)), w_tmp_2);
            dup(VReg4S(IDX(veps)), w_tmp_3);
        }

        XReg x_tmp_sp {x_tmp_2};
        mov(x_tmp_sp, XReg(IDX(rsp)));
        LDR_PARAM_TMP(N_nthr, eps);
        str(x_tmp_0, pre_ptr(x_tmp_sp, stack_off_N_nthr));
        LDR_PARAM_TMP(N_ithr, N_nthr);
        STR_PARAM_TMP(stack_off_N_ithr, stack_off_N_nthr);

        LDR_PARAM_TMP(src, N_ithr);
        STR_PARAM_TMP(stack_off_src, stack_off_N_ithr);

        LDR_PARAM_TMP(dst, src);
        STR_PARAM_TMP(stack_off_dst, stack_off_src);

        LDR_PARAM_TMP(diff_src, dst);
        STR_PARAM_TMP(stack_off_diff_src, stack_off_dst);

        LDR_PARAM_TMP(diff_dst, diff_src);
        STR_PARAM_TMP(stack_off_diff_dst, stack_off_diff_src);

        LDR_PARAM_TMP(ws, diff_dst);
        STR_PARAM_TMP(stack_off_ws, stack_off_diff_dst);

        LDR_PARAM_TMP(barrier, ws);
        STR_PARAM_TMP(stack_off_barrier, stack_off_ws);

        size_t tmpSize = PARAM_OFF(barrier);
        int32_t tmpStack = stack_off_barrier;

        if (is_spatial_thr_) {
            ldr(x_tmp_0,
                    pre_ptr(X_DEFAULT_ADDR,
                            PARAM_OFF(spat_size_loc) - tmpSize));
            STR_PARAM_TMP(stack_off_spat_size_loc, tmpStack);
            LDR_PARAM_TMP(S_s, spat_size_loc);
            STR_PARAM_TMP(stack_off_s_s, stack_off_spat_size_loc);
            LDR_PARAM_TMP(S_tail, S_s);
            STR_PARAM_TMP(stack_off_s_tail, stack_off_s_s);
            tmpSize = PARAM_OFF(S_tail);
            tmpStack = stack_off_s_tail;
        }
        if (is_c_padded()) {
            ldr(x_tmp_0,
                    pre_ptr(X_DEFAULT_ADDR, PARAM_OFF(is_cblk_tail) - tmpSize));
            STR_PARAM_TMP(stack_off_is_cblk_tail, tmpStack);
            tmpSize = PARAM_OFF(is_cblk_tail);
            tmpStack = stack_off_is_cblk_tail;
        }
        if (bdesc_->is_fwd()) {
            ldr(x_tmp_0, pre_ptr(X_DEFAULT_ADDR, PARAM_OFF(var) - tmpSize));
            mov(XReg(IDX(reg_var)), x_tmp_0);
        } else {
            ldr(x_tmp_0,
                    pre_ptr(X_DEFAULT_ADDR,
                            PARAM_OFF(diff_scale_shift) - tmpSize));
            STR_PARAM_TMP(stack_off_diff_scale_shift, tmpStack);
            LDR_PARAM_TMP(var, diff_scale_shift);
            mov(XReg(IDX(reg_var)), x_tmp_0);
        }
#undef LDR_PARAM
#undef LDR_PARAM_TMP
#undef STR_PARAM_TMP
#undef PARAM_OFF
    }

    void prepare_tail_mask_avx512_common() {
        if (!is_c_padded()) return;

        const int tail = bdesc_->C() % (int)(vlen / sizeof(float));
        uint32_t idx = IDX(ktail_mask);
        switch (tail) {
            case 16: ptrue(PRegS(idx), VL16); break;
            case 8: ptrue(PRegS(idx), VL8); break;
            case 7: ptrue(PRegS(idx), VL7); break;
            case 6: ptrue(PRegS(idx), VL6); break;
            case 5: ptrue(PRegS(idx), VL5); break;
            case 4: ptrue(PRegS(idx), VL4); break;
            case 3: ptrue(PRegS(idx), VL3); break;
            case 2: ptrue(PRegS(idx), VL2); break;
            case 1: ptrue(PRegS(idx), VL1); break;
            default:
                index(z_tmp0.s, 1, 1);
                cmple(PRegS(idx), p_512 / T_z, z_tmp0.s, tail);
                break;
        }
    }

    void prepare_relu() {
        with_relu = bdesc_->is_fwd()
                ? bdesc_->with_relu_post_op() || bdesc_->fuse_norm_relu()
                : bdesc_->fuse_norm_relu();
        with_relu_inf_only = with_relu && bdesc_->is_fwd()
                && !(bdesc_->fuse_norm_relu() && bdesc_->is_training());

        vzero = bdesc_->is_fwd() ? vdiff_beta : vbeta;
        if (with_relu) {
            eor(ZRegD(IDX(vzero)), ZRegD(IDX(vzero)), ZRegD(IDX(vzero)));
        }
    }

    void fwd_process_relu_avx512_common(ZReg vdst, int offt = 0) {
        if (is_nspc_)
            lsr(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    bit_shift() % 64);
        else
            lsr(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), bit_shift() % 64);

        fcmlt(PRegS(IDX(kstore_mask)), P_ALL_ONE / T_z, ZRegS(IDX(vzero)),
                ZRegS(IDX(vdst)));

        PRegB p_mask(IDX(kstore_mask));
        if (is_nspc_)
            add(x_tmp_1, XReg(IDX(reg_ws)), XReg(IDX(reg_soff_nspc)));
        else
            add(x_tmp_1, XReg(IDX(reg_ws)), XReg(IDX(reg_soff)));
        if (offt / (1 << bit_shift()))
            add_imm(x_tmp_1, x_tmp_1, offt / (1 << bit_shift()), x_tmp_0);
        uzp1(p_tmp0.b, p_mask, p_mask);
        uzp1(p_tmp0.b, p_tmp0.b, p_tmp0.b);
        sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
        str(p_tmp0, ptr(X_TRANSLATOR_STACK));
        ldurh(w_tmp_0, ptr(X_TRANSLATOR_STACK));
        add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
        strh(w_tmp_0, ptr(x_tmp_1));

        sel(ZRegS(IDX(vdst)), PReg(IDX(kstore_mask)) / T_m, ZRegS(IDX(vdst)),
                ZRegS(IDX(vzero)));

        if (is_nspc_)
            lsl(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    bit_shift() % 64);
        else
            lsl(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), bit_shift() % 64);
    }

    void bwd_process_relu_avx512_common(ZReg vdiff_dst, int offt = 0) {
        PReg p_mask(IDX(kstore_mask));
        if (is_nspc_) {
            lsr(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    bit_shift() % 64);
            add(x_tmp_1, XReg(IDX(reg_ws)), XReg(IDX(reg_soff_nspc)));
        } else {
            lsr(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), bit_shift() % 64);
            add(x_tmp_1, XReg(IDX(reg_ws)), XReg(IDX(reg_soff)));
        }
        if (offt / (1 << bit_shift()))
            add_imm(x_tmp_1, x_tmp_1, offt / (1 << bit_shift()), x_tmp_0);

        sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
        ldurh(w_tmp_0, ptr(x_tmp_1));
        strh(w_tmp_0, ptr(X_TRANSLATOR_STACK));
        ldr(p_mask, ptr(X_TRANSLATOR_STACK));
        zip1(p_mask.b, p_mask.b, p_mask.b);
        zip1(p_mask.b, p_mask.b, p_mask.b);
        add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);

        not_(p_tmp0.b, P_ALL_ONE / T_z, PRegB(IDX(kstore_mask)));
        mov(ZRegD(IDX(vdiff_dst)), ZRegD(IDX(vdiff_dst)));
        mov(ZRegS(IDX(vdiff_dst)), p_tmp0 / T_m, 0);

        if (is_nspc_)
            lsl(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    bit_shift() % 64);
        else
            lsl(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), bit_shift() % 64);
    }

    void uni_load_spat_data(const ZReg &z, const XReg &x) {
        if (is_bf16_) {
#if 0
                constexpr bool isAvx2 = isa == avx2;
                const typename std::conditional<isAvx2, Xmm, Ymm>::type
                        dst_reg {src.getIdx()};
                const typename std::conditional<isAvx2, Ymm, Zmm>::type
                        src_reg {src.getIdx()};

                // convert f32 output to bf16
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(dst_reg, src_reg);
                else
                    bf16_emu_->vcvtneps2bf16(dst_reg, src_reg);

                vmovdqu16(dst.getAddress(), dst_reg);
#endif // #if 0
        } else {
            ldr(z, ptr(x));
        }
    }

    void uni_store_spat_data(const XReg &x, const ZReg &z) {
        if (is_bf16_) {
#if 0
                // convert bf16 input to f32
                vpmovzxwd(ZReg(dst.getIdx()), src.getAddress());
                vpslld(ZReg(dst.getIdx()), ZReg(dst.getIdx()), 0x10);
#endif // #if 0
        } else {
            str(z, ptr(x));
        }
    }

    void jump_check(const Label &l_no_mask) {
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_is_cblk_tail, x_tmp_1);
        ldr(XReg(IDX(reg_tmp)), ptr(x_tmp_0));
        cmp(XReg(IDX(reg_tmp)), 0);
        b(EQ, l_no_mask);

        add_imm(x_tmp_0, XReg(IDX(reg_coff)), vlen, x_tmp_1);
        mov(XReg(IDX(reg_tmp)), x_tmp_0);
        cmp(XReg(IDX(reg_tmp)), XReg(IDX(reg_coff_max)));
        b(LT, l_no_mask);
    }

    void uni_load_maybe_tail(const ZReg &z, const XReg &x) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            jump_check(l_no_mask);
            ld1w(z.s, ktail_mask / T_z, ptr(x));
        }
        L(l_no_mask);
        ldr(z, ptr(x));
        L(l_ret);
    }

    void uni_store_maybe_tail(const XReg &x, const ZReg &z) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            jump_check(l_no_mask);
            st1w(z.s, ktail_mask / T_z, ptr(x));
        }
        L(l_no_mask);
        str(z, ptr(x));
        L(l_ret);
    }

    void uni_vdivps_aarch64(const VReg &dst, const VReg &src, const VReg &src2,
            const PReg &pred) {
        (void)pred;
        fdiv(VReg4S(IDX(dst)), VReg4S(IDX(src)), VReg4S(IDX(src2)));
    }

#if 0 
    void uni_vdivps_aarch64(const ZReg &dst, const ZReg &src, const ZReg &src2,
            const PReg &pred) {
        uni_vdivps_aarch64(
                ZReg(dst.getIdx()), ZReg(src.getIdx()), ZReg(src2.getIdx()), pred);
    }
#endif // #if 0

    void uni_vdivps_aarch64(const ZReg &dst, const ZReg &src, const ZReg &src2,
            const PReg &pred) {
        uint32_t dstIdx = IDX(dst);
        uint32_t srcIdx = IDX(src);
        uint32_t src2Idx = IDX(src2);

        if (dstIdx == src2Idx) {
            mov(z_tmp0.d, ZRegD(src2Idx));
            mov(ZRegD(dstIdx), ZRegD(srcIdx));
            fdiv(ZRegS(dstIdx), pred / T_m, z_tmp0.s);
        } else if (dstIdx == srcIdx) {
            fdiv(ZRegS(dstIdx), pred / T_m, ZRegS(src2Idx));
        } else {
            mov(ZRegD(dstIdx), ZRegD(srcIdx));
            fdiv(ZRegS(dstIdx), pred / T_m, ZRegS(src2Idx));
        }
    }

    void uni_vsqrtps_aarch64(
            const VReg &dst, const VReg &src, const PReg &pred) {
        (void)pred;
        fsqrt(VReg4S(IDX(dst)), VReg4S(IDX(src)));
    }

#if 0 
    void uni_vsqrtps_aarch64(
            const ZReg &dst, const ZReg &src, const PReg &pred) {
        uni_vsqrtps_aarch64(ZReg(dst.getIdx()), ZReg(src.getIdx()), pred);
    }
#endif // #if 0

    void uni_vsqrtps_aarch64(
            const ZReg &dst, const ZReg &src, const PReg &pred) {
        fsqrt(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)));
    }

    void uni_vaddps_unpredicate_aarch64(
            const VReg &dst, const VReg &src, const VReg &src2) {
        fadd(VReg4S(IDX(dst)), VReg4S(IDX(src)), VReg4S(IDX(src2)));
    }

#if 0 
    void uni_vaddps_unpredicate_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2) {
        uni_vaddps_unpredicate_aarch64(
                ZReg(dst.getIdx()), ZReg(src.getIdx()), ZReg(src2.getIdx()));
    }
#endif // #if 0

    void uni_vaddps_unpredicate_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2) {
        fadd(ZRegS(IDX(dst)), ZRegS(IDX(src)), ZRegS(IDX(src2)));
    }

    void uni_vfnmadd231ps_aarch64(const VReg &dst, const VReg &src,
            const VReg &src2, const PReg &pred) {
        fmls(VReg4S(IDX(dst)), VReg4S(IDX(src)), VReg4S(IDX(src2)));
    }

#if 0 
    void uni_vfnmadd231ps_aarch64(const ZReg &dst, const ZReg &src,
            const ZReg &src2, const PReg &pred) {
        uni_vfnmadd231ps_aarch64(
                ZReg(dst.getIdx()), ZReg(src.getIdx()), ZReg(src2.getIdx()), pred);
    }
#endif // #if 0

    void uni_vfnmadd231ps_aarch64(const ZReg &dst, const ZReg &src,
            const ZReg &src2, const PReg &pred) {
        fmls(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)), ZRegS(IDX(src2)));
    }

    void uni_vfmadd231ps_aarch64(
            const VReg &dst, const VReg &src, const VReg &src2, PReg &pred) {
        fmla(VReg4S(IDX(dst)), VReg4S(IDX(src)), VReg4S(IDX(src2)));
    }

#if 0 
    void uni_vfmadd231ps_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2, PReg &pred) {
        fmla(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)),
                ZRegS(IDX(src2)));
    }
#endif // #if 0

    void uni_vfmadd231ps_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2, PReg &pred) {
        fmla(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)), ZRegS(IDX(src2)));
    }

    void uni_vfmadd213ps_aarch64(
            const VReg &dst, const VReg &src, const VReg &src2, PReg &pred) {
        fmad(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)), ZRegS(IDX(src2)));
    }

#if 0
    void uni_vfmadd213ps_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2, PReg &pred) {
        fmad(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)),
                ZRegS(IDX(src2)));
    }
#endif // #if 0

    void uni_vfmadd213ps_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2, PReg &pred) {
        fmad(ZRegS(IDX(dst)), pred / T_m, ZRegS(IDX(src)), ZRegS(IDX(src2)));
    }

    void uni_vmovntps_aarch64(const XReg &base, const XReg &off, const int disp,
            const VReg &v, const PReg &p512, const PReg &p256,
            const PReg &p128) {
        add(X_DEFAULT_ADDR, XReg(IDX(base)), XReg(IDX(off)));
        if (disp != 0) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, disp, x_tmp_0);

        stnt1w(ZRegS(IDX(v)), p128, ptr(X_DEFAULT_ADDR));
    }

#if 0
    void uni_vmovntps_aarch64(const XReg &base, const XReg &off,
            const int disp, const ZReg &v, const PReg &p512,
            const PReg &p256, const PReg &p128) {
        add(X_DEFAULT_ADDR, XReg(IDX(base)), XReg(IDX(off)));
        if (disp != 0)
            add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, disp, x_tmp_0);

        stnt1w(ZRegS(IDX(v)), p256, ptr(X_DEFAULT_ADDR));
    }
#endif // #if 0

    void uni_vmovntps_aarch64(const XReg &base, const XReg &off, const int disp,
            const ZReg &v, const PReg &p512, const PReg &p256,
            const PReg &p128) {
        add(X_DEFAULT_ADDR, XReg(IDX(base)), XReg(IDX(off)));
        if (disp != 0) add_imm(X_DEFAULT_ADDR, X_DEFAULT_ADDR, disp, x_tmp_0);

        stnt1w(ZRegS(IDX(v)), p_512, ptr(X_DEFAULT_ADDR));
    }

    void uni_vmovups_aarch64(const VReg &v, const XReg &x) {
        ld1(VReg4S(IDX(v)), ptr(x));
    }

#if 0
    void uni_vmovups_aarch64(const ZReg &y, const XReg &x) {
        ld1w(ZRegS(IDX(y)), p_lsb_256, ptr(x));
    }
#endif // #if 0

    void uni_vmovups_aarch64(const ZReg &z, const XReg &x) {
        ldr(ZReg(IDX(z)), ptr(x));
    }

    void uni_vmovups_aarch64(const XReg &x, const VReg &v) {
        st1(VReg4S(IDX(v)), ptr(x));
    }

#if 0
    void uni_vmovups_aarch64(const XReg &x, const ZReg &y) {
        str1w(ZRegS(IDX(y)), p_lsb_256, ptr(x));
    }
#endif // #if 0

    void uni_vmovups_aarch64(const XReg &x, const ZReg &z) { str(z, ptr(x)); }

    void uni_vmaxps_aarch64(
            const VReg &dst, const VReg &src, const VReg &src2) {
        fmaxnm(VReg4S(IDX(dst)), VReg4S(IDX(src)), VReg4S(IDX(src2)));
        fmax(VReg4S(IDX(dst)), VReg4S(IDX(dst)), VReg4S(IDX(src2)));
    }
#if 0
    void uni_vmaxps_aarch64(const ZReg &dst, const ZReg &src, const ZReg &src2) {
        mov(z_tmp0.d, ZRegD(IDX(src2)));
        fmaxnm(z_tmp0.s, p_512, ZRegS(IDX(src)));
        fmax(z_tmp0.s, p_512, ZRegS(IDX(src)));
        mov(ZRegD(IDX(dst)), z_tmp0.d);
        mov(ZRegS(IDX(dst)), P_MSB_256 / T_m, 0);
    }
#endif // #if 0

    void uni_vmaxps_aarch64(
            const ZReg &dst, const ZReg &src, const ZReg &src2) {
        mov(z_tmp0.d, ZRegD(IDX(src2)));
        fmaxnm(z_tmp0.s, p_512, ZRegS(IDX(src)));
        fmax(z_tmp0.s, p_512, ZRegS(IDX(src)));
        mov(ZRegD(IDX(dst)), z_tmp0.d);
    }

    void barrier() {
        add_imm(x_tmp_1, XReg(IDX(rsp)), (int)stack_off_N_nthr, x_tmp_0);
        ldr(XReg(IDX(reg_nnthr)), ptr(x_tmp_1));
        add_imm(x_tmp_1, XReg(IDX(rsp)), (int)stack_off_barrier, x_tmp_0);
        ldr(XReg(IDX(reg_bar)), ptr(x_tmp_1));
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    XReg mean_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_mean)), XReg(IDX(reg_coff)));
        if (offt) add_imm(x_tmp_2, x_tmp_2, offt, x_tmp_1);
        return x_tmp_2;
    }

    XReg var_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_var)), XReg(IDX(reg_coff)));
        if (offt) add_imm(x_tmp_2, x_tmp_2, offt, x_tmp_1);
        return x_tmp_2;
    }

    XReg diff_gamma_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_diff_scale_shift)), XReg(IDX(reg_coff)));
        if (offt) add_imm(x_tmp_2, x_tmp_2, offt, x_tmp_1);
        return x_tmp_2;
    }

    XReg diff_beta_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_diff_scale_shift)), XReg(IDX(reg_coff)));
        if (offt || chan_data_offt)
            add_imm(x_tmp_2, x_tmp_2, offt + chan_data_offt, x_tmp_1);
        return x_tmp_2;
    }

    XReg gamma_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_scale_shift)), XReg(IDX(reg_coff)));
        if (offt) add_imm(x_tmp_2, x_tmp_2, offt, x_tmp_1);
        return x_tmp_2;
    }

    XReg beta_ptr(size_t offt = 0) {
        add(x_tmp_2, XReg(IDX(reg_scale_shift)), XReg(IDX(reg_coff)));
        if (offt || chan_data_offt)
            add_imm(x_tmp_2, x_tmp_2, offt + chan_data_offt, x_tmp_1);
        return x_tmp_2;
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs, init_t init,
            body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;
        for (size_t i = 0; i < num_active_regs; i++)
            init(i);
        if (loop_unroll) {
            if (is_spatial_thr_) {
                mov_imm(x_tmp_0, (int)stack_off_spat_size_loc);
                ldr(XReg(IDX(reg_ctr)), ptr(x_tmp_0));
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_s, x_tmp_1);
                ldr(x_tmp_0, ptr(x_tmp_0));
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), x_tmp_0);
            } else {
                mov_imm(XReg(IDX(reg_ctr)), (int)loop_unroll);
            }
            Label label;
            L(label);
            {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }
                add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        (int)factor * spat_step, x_tmp_0);
                sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), (int)factor,
                        x_tmp_0);
                cbnz(XReg(IDX(reg_ctr)), label);
            }
            if (is_spatial_thr_) {
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_tail,
                        x_tmp_1);
                ldr(x_tmp_0, ptr(x_tmp_0));
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), x_tmp_0);
            }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail) {
            add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                    (int)loop_tail * spat_step, x_tmp_0);
        }

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label);
        {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(ZReg(0), x_tmp_0);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        if (base_reg)
                            eor(ZRegD((uint32_t)base_reg * 2),
                                    ZRegD((uint32_t)base_reg * 2),
                                    ZRegD((uint32_t)base_reg * 2));
                    },
                    [=](size_t base_reg, size_t i) {
                        ZReg v0 = ZReg(base_reg * 2 + 0);
                        ZReg v1 = ZReg(base_reg * 2 + 1);
                        size_t offt = i * vlen_spat_data_;
                        add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_load_spat_data(v1, x_tmp_0);
                        uni_vaddps_unpredicate_aarch64(v0, v0, v1);
                        add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt || t0_pf_offt)
                            add_imm(x_tmp_1, x_tmp_0, offt + t0_pf_offt,
                                    x_tmp_1);
                        prfm(PLDL1KEEP, ptr(x_tmp_1));
                        if (offt || t1_pf_offt)
                            add_imm(x_tmp_1, x_tmp_0, offt + t1_pf_offt,
                                    x_tmp_1);
                        prfm(PLDL2KEEP, ptr(x_tmp_1));
                    },
                    [=](size_t base_reg) {
                        ZReg b = ZReg(0);
                        ZReg v = ZReg(base_reg * 2);
                        if (base_reg) uni_vaddps_unpredicate_aarch64(b, b, v);
                    });
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));

            add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen, x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(LT, ch_label);
        }
    }

    void mean_variance_nspc(
            const int num_ch_blks, int num_spat_pts, bool compute_mean) {

        auto mean_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff_nspc)));
                    if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                    uni_load_spat_data(ZReg(sp_idx), x_tmp_0);

                    uni_vaddps_unpredicate_aarch64(
                            ZReg(ch_idx), ZReg(ch_idx), ZReg(sp_idx++));

                    offt += vlen_spat_data_;
                }
                add_imm(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                        (int)spat_step, x_tmp_0);
            }
        };

        auto variance_compute = [=](int num_ch_blks, int num_spat_pts) {
            int sp_idx = num_ch_blks;
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                int coff = 0, offt = 0;
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    uni_load_maybe_tail(vmean, mean_ptr(coff));

                    add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff_nspc)));
                    if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                    uni_load_spat_data(ZReg(sp_idx), x_tmp_0);

                    fsub(ZRegS(30), ZRegS(IDX(vmean)),
                            ZRegS((uint32_t)sp_idx++));
                    uni_vfmadd231ps_aarch64(
                            ZReg(ch_idx), ZReg(30), ZReg(30), p_512);

                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add_imm(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                        (int)spat_step, x_tmp_0);
            }
        };

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen) {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(ZReg(idx), x_tmp_0);
        }

        eor(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                XReg(IDX(reg_soff_nspc)));

        if (is_spatial_thr_) {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_spat_size_loc,
                    x_tmp_1);
            ldr(XReg(IDX(reg_ctr)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_s, x_tmp_1);
            ldr(x_tmp_0, ptr(x_tmp_0));
            add(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)), x_tmp_0);

            // TODO: need a better heuristic for num_spat_pts
            num_spat_pts = 1;
        } else {
            mov_imm(XReg(IDX(reg_ctr)), (int)spat_size);
            num_spat_pts = nstl::min((size_t)num_spat_pts, spat_size);
            // TODO: unroll by spatial
            if (spat_size % num_spat_pts != 0) num_spat_pts = 1;
        }

        Label spatial;
        L(spatial);
        {
            compute_mean ? mean_compute(num_ch_blks, num_spat_pts)
                         : variance_compute(num_ch_blks, num_spat_pts);
            sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), num_spat_pts,
                    x_tmp_0);
            cbnz(XReg(IDX(reg_ctr)), spatial);
        }

        for (int idx = 0, offt = 0; idx < num_ch_blks; ++idx, offt += vlen) {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(x_tmp_0, ZReg(idx));
        }
    }

    void forward_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            /* Overwritten during mean and variance computation */
            eor(ZRegD(IDX(vzero)), ZRegD(IDX(vzero)), ZRegD(IDX(vzero)));

            eor(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    XReg(IDX(reg_soff_nspc)));

            if (is_spatial_thr_) {
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_spat_size_loc,
                        x_tmp_1);
                ldr(XReg(IDX(reg_ctr)), ptr(x_tmp_0));
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_s, x_tmp_1);
                ldr(x_tmp_0, ptr(x_tmp_0));
                add(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                        x_tmp_0);
            } else {
                mov_imm(XReg(IDX(reg_ctr)), spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    uni_load_maybe_tail(vmean, mean_ptr(coff));
                    uni_load_maybe_tail(vsqrtvar, var_ptr(coff));
                    uni_vaddps_unpredicate_aarch64(vsqrtvar, vsqrtvar, veps);
                    uni_vsqrtps_aarch64(vsqrtvar, vsqrtvar, p_512);

                    if (bdesc_->use_scaleshift()) {
                        uni_load_maybe_tail(vgamma, gamma_ptr(coff));
                        uni_load_maybe_tail(vbeta, beta_ptr(coff));
                    }

                    ZReg vscale = bdesc_->use_scaleshift() ? vgamma : vone;
                    ZReg vdiv = bdesc_->use_scaleshift() ? vgamma : vsqrtvar;

                    uni_vdivps_aarch64(vdiv, vscale, vsqrtvar, p_512);

                    add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff_nspc)));
                    if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                    uni_load_spat_data(ZReg(idx), x_tmp_0);

                    fsub(ZRegS((uint32_t)idx), ZRegS((uint32_t)idx),
                            ZRegS(IDX(vmean)));

                    if (bdesc_->use_scaleshift()) { // --flags=S
                        uni_vfmadd213ps_aarch64(
                                ZReg(idx), vgamma, vbeta, p_512);
                    } else {
                        fmul(ZRegS((uint32_t)idx), ZRegS((uint32_t)idx),
                                ZRegS(IDX(vsqrtvar)));
                    }

                    if (with_relu_inf_only) { // --attr=post_ops='relu'
                        uni_vmaxps_aarch64(ZReg(idx), ZReg(idx), vzero);
                    } else if (with_relu) { // --flags=R
                        fwd_process_relu_avx512_common(ZReg(idx));
                    }

                    if (stream_store_allowed) {
                        uni_vmovntps_aarch64(reg_dst, reg_soff_nspc, offt,
                                ZReg(idx), p_512, p_lsb_256, p_lsb_128);
                    } else {
                        add(x_tmp_0, XReg(IDX(reg_dst)),
                                XReg(IDX(reg_soff_nspc)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_store_spat_data(x_tmp_0, ZReg(idx));
                    }

                    add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), 2, x_tmp_0);
                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add_imm(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                        (int)spat_step, x_tmp_0);
                sub_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), 2 * num_ch_blks,
                        x_tmp_0);
                sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), num_spat_pts,
                        x_tmp_0);
                cbnz(XReg(IDX(reg_ctr)), spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            cmp(XReg(IDX(reg_dst)), vlen - 1);
            cbnz(XReg(IDX(reg_dst)), normal_store);
            compute(true);
            b(normal_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void compute_mean_variance_nspc(bool compute_mean = true) {
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        mov(XReg(IDX(reg_coff_max_fwd_copy)), XReg(IDX(reg_coff_max)));

        Label ch_unroll_label[5];
        const int max_ch_unroll = 4;
        //                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll, sp_idx = 1; ch_idx > 0;
                --ch_idx, ++sp_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                mov_imm(x_tmp_0, vlen * ch_blk_size);
                cmp(XReg(IDX(reg_coff_max)), x_tmp_0);
                b(LT, ch_unroll_label[ch_idx - 1]);

                const int spat_blk_size = (1 << sp_idx);
                mean_variance_nspc(ch_blk_size, spat_blk_size, compute_mean);

                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)),
                        vlen * ch_blk_size, x_tmp_0);

                sub_imm(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)),
                        vlen * ch_blk_size, x_tmp_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max_fwd_copy)));

        //if (is_bf16_) shr(reg_coff_max, 1);
        sub(XReg(IDX(reg_src)), XReg(IDX(reg_src)), XReg(IDX(reg_coff_max)));
        //if (is_bf16_) shl(reg_coff_max, 1);
    }

    void var_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(ZReg(0), x_tmp_0);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [=](size_t base_reg) {
                        if (base_reg > 0)
                            eor(ZRegD((uint32_t)base_reg * 3),
                                    ZRegD((uint32_t)base_reg * 3),
                                    ZRegD((uint32_t)base_reg * 3));
                    },
                    [=](size_t base_reg, size_t i) {
                        ZReg v = ZReg(3 * base_reg);
                        ZReg vtmp0 = ZReg(3 * base_reg + 1);
                        ZReg vtmp1 = ZReg(3 * base_reg + 2);
                        size_t offt = i * vlen_spat_data_;
                        add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_load_spat_data(vtmp0, x_tmp_0);
                        fsub(ZRegS(IDX(vtmp1)), ZRegS(IDX(vmean)),
                                ZRegS(IDX(vtmp0)));
                        uni_vfmadd231ps_aarch64(v, vtmp1, vtmp1, p_512);

                        add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt || t0_pf_offt)
                            add_imm(x_tmp_1, x_tmp_0, offt + t0_pf_offt,
                                    x_tmp_1);
                        prfm(PLDL1KEEP, ptr(x_tmp_1));

                        if (offt || t1_pf_offt)
                            add_imm(x_tmp_1, x_tmp_0, offt + t1_pf_offt,
                                    x_tmp_1);
                        prfm(PLDL2KEEP, ptr(x_tmp_1));
                    },
                    [=](size_t base_reg) {
                        ZReg b = ZReg(0);
                        ZReg v = ZReg(base_reg * 3);
                        if (base_reg) uni_vaddps_unpredicate_aarch64(b, b, v);
                    });
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));
            add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen, x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(LT, ch_label);
        }
    }

    void compute_mean_variance() {
        eor(ZRegD(0), ZRegD(0), ZRegD(0));
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        Label zero_rbuf;
        L(zero_rbuf);
        {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));
            add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen, x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(NE, zero_rbuf);
        }

        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
        ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));

        eor(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), XReg(IDX(reg_soff)));
        Label mean_spatial;
        L(mean_spatial);
        {
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));

            is_nspc_ ? compute_mean_variance_nspc() : mean_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (mb_offt) {
                    add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)), mb_offt,
                            x_tmp_0);
                    add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), mb_offt,
                            x_tmp_0);
                }
            } else {
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        XReg(IDX(reg_mb_stride_Bc)));
            }

            cmp(XReg(IDX(reg_soff)), XReg(IDX(reg_soff_max)));
            b(LT, mean_spatial);
        }

        if (is_nspc_) {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
            ldr(XReg(IDX(reg_src)), ptr(x_tmp_0)); // comeback
        }

        Label no_mean_reduction;
        barrier();
        {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_ithr, x_tmp_1);
            ldr(XReg(IDX(reg_tmp)), ptr(x_tmp_0));
            cmp(XReg(IDX(reg_tmp)), 0);
            b(NE, no_mean_reduction);
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_nthr, x_tmp_1);
            ldr(XReg(IDX(reg_nnthr)), ptr(x_tmp_0));
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(XReg(IDX(reg_roff)), XReg(IDX(reg_coff)));
                eor(ZRegD(0), ZRegD(0), ZRegD(0));
                eor(ZRegD(1), ZRegD(1), ZRegD(1));
                mov(XReg(IDX(reg_ctr)), XReg(IDX(reg_nnthr)));
                Label mean_reduction_thrs;
                L(mean_reduction_thrs);
                {
                    add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_roff)));
		    if (isa == sve_512) {
                        ld1w(z_tmp0.s, p_512 / T_z, ptr(x_tmp_0));
                        fadd(ZRegS(1), ZRegS(1), z_tmp0.s);
#if 0
                    } else if (ZReg(1).isYMM()) {
                        ld1w(z_tmp0.s, p_lsb_256 / T_z,
                                ptr(x_tmp_0));
                        fadd(ZRegS(1), p_lsb_256 / T_z, z_tmp0.s);
#endif // #if 0
                    } else {
                        ldr(QReg(tmp_vec_idx[0]), ptr(x_tmp_0));
                        fadd(VReg4S(1), VReg4S(1), VReg4S(tmp_vec_idx[0]));
                    }

                    add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_roff)));
                    uni_vmovups_aarch64(x_tmp_0, ZReg(0));
                    add(XReg(IDX(reg_roff)), XReg(IDX(reg_roff)),
                            XReg(IDX(reg_coff_max)));
                    sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), 1, x_tmp_0);
                    cbnz(XReg(IDX(reg_ctr)), mean_reduction_thrs);
                }
		if (isa == sve_512)
                    fdiv(ZRegS(1), p_512 / T_m, ZRegS(IDX(vchan_size)));
#if 0
                else if (vchan_size.isYMM())
                    fdiv(ZRegS(1), p_lsb_256 / T_m,
                            ZRegS(IDX(vchan_size)));
#endif // #if 0
                else {
                    fdiv(VReg4S(1), VReg4S(1), VReg4S(IDX(vchan_size)));
                }
                uni_store_maybe_tail(mean_ptr(), ZReg(1));

		if (isa == sve_512)
                    add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                            x_tmp_0);
		else if (isa == asimd)
                    add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen / 2,
                            x_tmp_0);

                cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
                b(LT, mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        eor(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), XReg(IDX(reg_soff)));
        Label var_spatial;
        L(var_spatial);
        {
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));

            is_nspc_ ? compute_mean_variance_nspc(false) : var_channels();

            // Process next image
            if (is_nspc_ && mb_offt) {
                // Can use static offset since we comeback after spatial loop
                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), mb_offt,
                        x_tmp_0);
            } else {
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        XReg(IDX(reg_mb_stride_Bc)));
            }

            cmp(XReg(IDX(reg_soff)), XReg(IDX(reg_soff_max)));
            b(LT, var_spatial);
        }

        if (is_nspc_) {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
            ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
        }

        Label no_var_reduction;
        barrier();
        {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_ithr, x_tmp_1);
            ldr(XReg(IDX(reg_tmp)), ptr(x_tmp_0));
            cmp(XReg(IDX(reg_tmp)), 0);
            b(NE, no_var_reduction);

            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_nthr, x_tmp_1);
            ldr(XReg(IDX(reg_nnthr)), ptr(x_tmp_0));
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
            Label var_reduction_channels;
            L(var_reduction_channels);
            {
                mov(XReg(IDX(reg_roff)), XReg(IDX(reg_coff)));
                eor(ZRegD(1), ZRegD(1), ZRegD(1));
                mov(XReg(IDX(reg_ctr)), XReg(IDX(reg_nnthr)));
                Label var_reduction_thrs;
                L(var_reduction_thrs);
                { // TODO: unroll (?)
                    add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_roff)));
		    if (isa == sve_512) {
                        ld1w(z_tmp0.s, p_512 / T_z, ptr(x_tmp_0));
                        fadd(ZRegS(1), ZRegS(1), z_tmp0.s);
#if 0
                    } else if (ZReg(1).isYMM()) {
                        ld1w(z_tmp0.s, p_lsb_256 / T_z,
                                ptr(x_tmp_0));
                        fadd(ZRegS(1), p_lsb_256 / T_z, z_tmp0.s);
#endif // #if 0
                    } else {
                        ldr(QReg(tmp_vec_idx[0]), ptr(x_tmp_0));
                        fadd(VReg4S(1), VReg4S(1), VReg4S(tmp_vec_idx[0]));
                    }
                    add(XReg(IDX(reg_roff)), XReg(IDX(reg_roff)),
                            XReg(IDX(reg_coff_max)));
                    sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), 1, x_tmp_0);
                    cbnz(XReg(IDX(reg_ctr)), var_reduction_thrs);
                }
		if (isa == sve_512)
                    fdiv(ZRegS(1), p_512 / T_m, ZRegS(IDX(vchan_size)));
#if 0
                else if (vchan_size.isYMM())
                    fdiv(ZRegS(1), p_lsb_256 / T_m,
                            ZRegS(IDX(vchan_size)));
#endif // #if 0
                else {
                    fdiv(VReg4S(1), VReg4S(1), VReg4S(IDX(vchan_size)));
                }
                uni_store_maybe_tail(var_ptr(), ZReg(1));
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                        x_tmp_0);

                cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
                b(NE, var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            uni_load_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps_unpredicate_aarch64(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps_aarch64(vsqrtvar, vsqrtvar, p_512);

            if (bdesc_->use_scaleshift()) {
                uni_load_maybe_tail(vgamma, gamma_ptr());
                uni_load_maybe_tail(vbeta, beta_ptr());
            }

            ZReg vscale = bdesc_->use_scaleshift() ? vgamma : vone;
            ZReg vdiv = bdesc_->use_scaleshift() ? vgamma : vsqrtvar;

            uni_vdivps_aarch64(vdiv, vscale, vsqrtvar, p_512);

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            ZReg v = ZReg(base_reg);
                            size_t offt = i * vlen_spat_data_;
                            add(x_tmp_0, XReg(IDX(reg_src)),
                                    XReg(IDX(reg_soff)));
                            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                            uni_load_spat_data(v, x_tmp_0);
                            add(x_tmp_0, XReg(IDX(reg_src)),
                                    XReg(IDX(reg_soff)));
                            if (offt || t0_pf_offt)
                                add_imm(x_tmp_1, x_tmp_0, offt + t0_pf_offt,
                                        x_tmp_1);
                            prfm(PLDL1KEEP, ptr(x_tmp_1));

                            if (offt || t1_pf_offt)
                                add_imm(x_tmp_1, x_tmp_0, offt + t1_pf_offt,
                                        x_tmp_1);
                            prfm(PLDL2KEEP, ptr(x_tmp_1));
                            fsub(ZRegS(IDX(v)), ZRegS(IDX(v)),
                                    ZRegS(IDX(vmean)));
                            if (bdesc_->use_scaleshift()) {
                                uni_vfmadd213ps_aarch64(
                                        v, vgamma, vbeta, p_512);
                            } else {
                                fmul(ZRegS(IDX(v)), ZRegS(IDX(v)),
                                        ZRegS(IDX(vsqrtvar)));
                            }
                            if (with_relu_inf_only) {
                                uni_vmaxps_aarch64(v, v, vzero);
                            } else if (with_relu) {
                                if (isa == sve_512)
                                    fwd_process_relu_avx512_common(v, offt);
                            }
                            if (stream_store_allowed) {
                                uni_vmovntps_aarch64(reg_dst, reg_soff, offt, v,
                                        p_512, p_lsb_256, p_lsb_128);
                            } else {
                                add(x_tmp_0, XReg(IDX(reg_dst)),
                                        XReg(IDX(reg_soff)));
                                if (offt)
                                    add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                                uni_store_spat_data(x_tmp_0, v);
                            }
                        },
                        [](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                cmp(XReg(IDX(reg_dst)), vlen - 1);
                cbnz(XReg(IDX(reg_dst)), normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            if (vlen)
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                        x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(LT, ch_label);
        }
    }

    void forward_channels_nspc() {
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        mov(XReg(IDX(reg_coff_max_fwd_copy)), XReg(IDX(reg_coff_max)));

        Label ch_unroll_label[5];
        const int max_ch_unroll
                //                = is_bf16_ && !mayiuse(avx512_core_bf16) ? 3 : 4;
                = is_bf16_ ? 3 : 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                mov_imm(x_tmp_0, vlen * ch_blk_size);
                cmp(XReg(IDX(reg_coff_max)), x_tmp_0);
                b(LT, ch_unroll_label[ch_idx - 1]);

                forward_channels_nspc_compute(ch_blk_size);

                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);
                add_imm(XReg(IDX(reg_dst)), XReg(IDX(reg_dst)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);

                // advance mean_ptr() and var_ptr()
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)),
                        vlen * ch_blk_size, x_tmp_0);

                add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), 2 * ch_blk_size,
                        x_tmp_0);

                sub_imm(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)),
                        vlen * ch_blk_size, x_tmp_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max_fwd_copy)));

        //if (is_bf16_) shr(reg_coff_max, 1);
        sub(XReg(IDX(reg_src)), XReg(IDX(reg_src)), XReg(IDX(reg_coff_max)));
        sub(XReg(IDX(reg_dst)), XReg(IDX(reg_dst)), XReg(IDX(reg_coff_max)));
        //if (is_bf16_) shl(reg_coff_max, 1);

        lsr(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
        sub(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), XReg(IDX(reg_coff_max)));
        lsl(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
    }

    void forward() {
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
        ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_dst, x_tmp_1);
        ldr(XReg(IDX(reg_dst)), ptr(x_tmp_0));
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws, x_tmp_1);
        ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));

        eor(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), XReg(IDX(reg_soff)));
        Label dst_spatial;
        L(dst_spatial);
        {
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));

            is_nspc_ ? forward_channels_nspc() : forward_channels();

            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_dst)), XReg(IDX(reg_dst)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), ws_mb_offt,
                        x_tmp_0);
            } else {
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        XReg(IDX(reg_mb_stride_Bc)));
            }

            cmp(XReg(IDX(reg_soff)), XReg(IDX(reg_soff_max)));
            b(LT, dst_spatial);
        }

        if (is_nspc_) {
            // comeback
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
            ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_dst, x_tmp_1);
            ldr(XReg(IDX(reg_dst)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws, x_tmp_1);
            ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));
        }
    }

    void backward_sh_channels() {
        Label sh_channels;
        L(sh_channels);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(ZReg(0), x_tmp_0);
            add(x_tmp_0, XReg(IDX(reg_rbuf2)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(ZReg(1), x_tmp_0);
            spat_loop(
                    spat_size, 1, 1,
                    [=](size_t base_reg) {
                        if (base_reg > 0) {
                            for (int i = 0; i < 2; i++) {
                                ZReg v(base_reg * 5 + i);
                                eor(ZRegD(IDX(v)), ZRegD(IDX(v)),
                                        ZRegD(IDX(v)));
                            }
                        }
                    },
                    [=](size_t base_reg, size_t i) {
                        ZReg o0 = ZReg(base_reg * 5 + 0);
                        ZReg o1 = ZReg(base_reg * 5 + 1);
                        ZReg t1 = ZReg(base_reg * 5 + 2);
                        ZReg t2 = ZReg(base_reg * 5 + 3);
                        ZReg t3 = ZReg(base_reg * 5 + 4);
                        size_t offt = i * vlen_spat_data_;
                        add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_load_spat_data(t1, x_tmp_0);
                        add(x_tmp_0, XReg(IDX(reg_diff_dst)),
                                XReg(IDX(reg_soff)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_load_spat_data(t2, x_tmp_0);
                        if (with_relu) {
                            if (isa == sve_512)
                                bwd_process_relu_avx512_common(t2, offt);
                            else
                                assert(false);
                        }
                        fsub(ZRegS(IDX(t3)), ZRegS(IDX(vmean)), ZRegS(IDX(t1)));
                        uni_vfnmadd231ps_aarch64(o0, t3, t2, p_512);
                        uni_vaddps_unpredicate_aarch64(o1, o1, t2);
                        add(x_tmp_0, XReg(IDX(reg_diff_dst)),
                                XReg(IDX(reg_soff)));
                        add(x_tmp_1, XReg(IDX(reg_src)), XReg(IDX(reg_soff)));
                        if (offt || t0_pf_offt)
                            add_imm(x_tmp_2, x_tmp_0, offt + t0_pf_offt,
                                    x_tmp_2);
                        prfm(PLDL1KEEP, ptr(x_tmp_2));
                        if (offt || t0_pf_offt)
                            add_imm(x_tmp_2, x_tmp_1, offt + t0_pf_offt,
                                    x_tmp_2);
                        prfm(PLDL1KEEP, ptr(x_tmp_2));

                        if (offt || t1_pf_offt)
                            add_imm(x_tmp_2, x_tmp_0, offt + t1_pf_offt,
                                    x_tmp_2);
                        prfm(PLDL2KEEP, ptr(x_tmp_2));
                        if (offt || t1_pf_offt)
                            add_imm(x_tmp_2, x_tmp_1, offt + t1_pf_offt,
                                    x_tmp_2);
                        prfm(PLDL2KEEP, ptr(x_tmp_2));
                    },
                    [=](size_t base_reg) {
                        ZReg b0 = ZReg(0);
                        ZReg b1 = ZReg(1);
                        if (base_reg) {
                            uni_vaddps_unpredicate_aarch64(
                                    b0, b0, ZReg(base_reg * 5 + 0));
                            uni_vaddps_unpredicate_aarch64(
                                    b1, b1, ZReg(base_reg * 5 + 1));
                        }
                    });
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));
            add(x_tmp_0, XReg(IDX(reg_rbuf2)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(1));
            if (vlen)
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                        x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(LT, sh_channels);
        }
    }

    void backward_sh_channels_nspc_compute(const int num_ch_blks) {
        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(ZReg(idx++), x_tmp_0);
            add(x_tmp_0, XReg(IDX(reg_rbuf2)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(ZReg(idx++), x_tmp_0);
        }

        eor(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                XReg(IDX(reg_soff_nspc)));

        if (is_spatial_thr_) {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_spat_size_loc,
                    x_tmp_1);
            ldr(XReg(IDX(reg_ctr)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_s, x_tmp_1);
            ldr(XReg(IDX(reg_soff_nspc)), ptr(x_tmp_0));
        } else {
            mov_imm(XReg(IDX(reg_ctr)), spat_size);
        }

        // TODO: spatial blocking
        const int num_spat_pts = 1;

        Label spatial;
        L(spatial);
        {
            int coff = 0, offt = 0, sp_idx = 2 * num_ch_blks;
            for (int ch_idx = 0; ch_idx < 2 * num_ch_blks; ch_idx += 2) {
                uni_load_maybe_tail(vmean, mean_ptr(coff));

                add(x_tmp_0, XReg(IDX(reg_src)), XReg(IDX(reg_soff_nspc)));
                if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                uni_load_spat_data(ZReg(sp_idx), x_tmp_0);
                add(x_tmp_0, XReg(IDX(reg_diff_dst)), XReg(IDX(reg_soff_nspc)));
                if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                uni_load_spat_data(ZReg(sp_idx + 1), x_tmp_0);

                if (with_relu) {
                    if (isa == sve_512)
                        bwd_process_relu_avx512_common(ZReg(sp_idx + 1), offt);
                    else
                        assert(false);
                }

                fsub(ZRegS((uint32_t)sp_idx + 2), ZRegS(IDX(vmean)),
                        ZRegS((uint32_t)sp_idx));
                uni_vfnmadd231ps_aarch64(ZReg(ch_idx), ZReg(sp_idx + 2),
                        ZReg(sp_idx + 1), p_512);
                uni_vaddps_unpredicate_aarch64(
                        ZReg(ch_idx + 1), ZReg(ch_idx + 1), ZReg(sp_idx + 1));

                coff += vlen;
                offt += vlen_spat_data_;
                sp_idx += 3;
            }
            add_imm(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    spat_step, x_tmp_0);
            sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), num_spat_pts,
                    x_tmp_0);
            cbnz(XReg(IDX(reg_ctr)), spatial);
        }

        for (int idx = 0, offt = 0; idx < 2 * num_ch_blks; offt += vlen) {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(x_tmp_0, ZReg(idx++));
            add(x_tmp_0, XReg(IDX(reg_rbuf2)), XReg(IDX(reg_coff)));
            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
            uni_vmovups_aarch64(x_tmp_0, ZReg(idx++));
        }
    }

    void backward_sh_channels_nspc() {
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        mov(XReg(IDX(reg_coff_max_bwd_copy)), XReg(IDX(reg_coff_max)));

        Label ch_unroll_label[5];
        const int max_ch_unroll = is_bf16_ ? 1 : 3;
        //= is_bf16_ && !mayiuse(avx512_core_bf16) ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(XReg(IDX(reg_coff_max)), vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                backward_sh_channels_nspc_compute(ch_blk_size);

                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);
                add_imm(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);

                // advance mean_ptr() and var_ptr()
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)),
                        vlen * ch_blk_size, x_tmp_0);

                add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), 2 * ch_blk_size,
                        x_tmp_0);

                sub_imm(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)),
                        vlen * ch_blk_size, x_tmp_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max_bwd_copy)));
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_scale_shift,
                x_tmp_1);
        ldr(XReg(IDX(reg_diff_scale_shift)), ptr(x_tmp_0));

        //if (is_bf16_) shr(reg_coff_max, 1);
        sub(XReg(IDX(reg_src)), XReg(IDX(reg_src)), XReg(IDX(reg_coff_max)));
        sub(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                XReg(IDX(reg_coff_max)));
        //if (is_bf16_) shl(reg_coff_max, 1);

        if (with_relu) {
            lsr(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
            sub(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), XReg(IDX(reg_coff_max)));
            lsl(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
        }
    }

    void backward_diff_channels() {
        Label diff_channels;
        L(diff_channels);
        {
            uni_load_maybe_tail(vmean, mean_ptr());
            uni_load_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps_unpredicate_aarch64(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps_aarch64(vsqrtvar, vsqrtvar, p_512);
            uni_vdivps_aarch64(vsqrtvar, vone, vsqrtvar, p_512);
            if (bdesc_->use_scaleshift())
                uni_load_maybe_tail(vgamma, gamma_ptr());
            uni_load_maybe_tail(vdiff_gamma, diff_gamma_ptr());
            uni_load_maybe_tail(vdiff_beta, diff_beta_ptr());
            fmul(ZRegS(IDX(vdiff_gamma)), ZRegS(IDX(vdiff_gamma)),
                    ZRegS(IDX(vsqrtvar)));
            uni_vdivps_aarch64(vdiff_beta, vdiff_beta, vchan_size, p_512);
            uni_vdivps_aarch64(vdiff_gamma, vdiff_gamma, vchan_size, p_512);

            auto compute = [=](bool stream_store_allowed) {
                spat_loop(
                        spat_size, unroll_blocks, unroll_regs,
                        [=](size_t base_reg) { UNUSED(base_reg); },
                        [=](size_t base_reg, size_t i) {
                            ZReg v(base_reg * 2 + 0);
                            ZReg t(base_reg * 2 + 1);
                            ZReg t1(base_reg * 2 + 2);
                            size_t offt = i * vlen_spat_data_;
                            add(x_tmp_0, XReg(IDX(reg_diff_dst)),
                                    XReg(IDX(reg_soff)));
                            if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                            uni_load_spat_data(v, x_tmp_0);
                            if (with_relu) {
                                if (isa == sve_512)
                                    bwd_process_relu_avx512_common(v, offt);
                                else
                                    assert(false);
                            }
                            if (!bdesc_->use_global_stats()) {
                                fsub(ZRegS(IDX(v)), ZRegS(IDX(v)),
                                        ZRegS(IDX(vdiff_beta)));
                                add(x_tmp_0, XReg(IDX(reg_src)),
                                        XReg(IDX(reg_soff)));
                                if (offt)
                                    add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                                uni_load_spat_data(t, x_tmp_0);
                                fsub(ZRegS(IDX(t)), ZRegS(IDX(vmean)),
                                        ZRegS(IDX(t)));
                                fmul(ZRegS(IDX(t)), ZRegS(IDX(t)),
                                        ZRegS(IDX(vdiff_gamma)));
                                uni_vaddps_unpredicate_aarch64(v, v, t);
                            }
                            fmul(ZRegS(IDX(v)), ZRegS(IDX(v)),
                                    ZRegS(IDX(vsqrtvar)));
                            if (bdesc_->use_scaleshift()) {
                                fmul(ZRegS(IDX(v)), ZRegS(IDX(v)),
                                        ZRegS(IDX(vgamma)));
                            }
                            if (stream_store_allowed) {
                                uni_vmovntps_aarch64(reg_diff_src, reg_soff,
                                        offt, v, p_512, p_lsb_256, p_lsb_128);
                            } else {
                                add(x_tmp_0, XReg(IDX(reg_diff_src)),
                                        XReg(IDX(reg_soff)));
                                if (offt)
                                    add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                                uni_store_spat_data(x_tmp_0, v);
                            }
                            add(x_tmp_0, XReg(IDX(reg_diff_dst)),
                                    XReg(IDX(reg_soff)));
                            add(x_tmp_1, XReg(IDX(reg_src)),
                                    XReg(IDX(reg_soff)));
                            if (offt || t0_pf_offt)
                                add_imm(x_tmp_2, x_tmp_0, offt + t0_pf_offt,
                                        x_tmp_2);
                            prfm(PLDL1KEEP, ptr(x_tmp_2));
                            if (offt || t0_pf_offt)
                                add_imm(x_tmp_2, x_tmp_1, offt + t0_pf_offt,
                                        x_tmp_2);
                            prfm(PLDL1KEEP, ptr(x_tmp_2));

                            if (offt || t1_pf_offt)
                                add_imm(x_tmp_2, x_tmp_0, offt + t1_pf_offt,
                                        x_tmp_2);
                            prfm(PLDL2KEEP, ptr(x_tmp_2));
                            if (offt || t1_pf_offt)
                                add_imm(x_tmp_2, x_tmp_1, offt + t1_pf_offt,
                                        x_tmp_2);
                            prfm(PLDL2KEEP, ptr(x_tmp_2));
                        },
                        [=](size_t base_reg) { UNUSED(base_reg); });
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                cmp(XReg(IDX(reg_diff_src)), vlen - 1);
                cbnz(XReg(IDX(reg_diff_src)), normal_store);
                compute(true);
                b(end_store);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            if (vlen)
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                        x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(LT, diff_channels);
        }
    }

    void backward_diff_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [=](bool stream_store_allowed) {
            eor(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                    XReg(IDX(reg_soff_nspc)));
            if (is_spatial_thr_) {
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_spat_size_loc,
                        x_tmp_1);
                ldr(XReg(IDX(reg_ctr)), ptr(x_tmp_0));
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_s_s, x_tmp_1);
                ldr(XReg(IDX(reg_soff_nspc)), ptr(x_tmp_0));
            } else {
                mov_imm(XReg(IDX(reg_ctr)), spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            Label spatial;
            L(spatial);
            {
                int coff = 0, offt = 0;
                for (int idx = 0; idx < 3 * num_ch_blks; idx += 3) {
                    uni_load_maybe_tail(vmean, mean_ptr(coff));
                    uni_load_maybe_tail(vsqrtvar, var_ptr(coff));

                    uni_vaddps_unpredicate_aarch64(vsqrtvar, vsqrtvar, veps);
                    uni_vsqrtps_aarch64(vsqrtvar, vsqrtvar, p_512);
                    uni_vdivps_aarch64(vsqrtvar, vone, vsqrtvar, p_512);

                    if (bdesc_->use_scaleshift())
                        uni_load_maybe_tail(vgamma, gamma_ptr(coff));

                    add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws_off_copy,
                            x_tmp_1);
                    str(XReg(IDX(reg_ws)), ptr(x_tmp_0));
                    add_imm(x_tmp_0, XReg(IDX(rsp)),
                            (int)stack_off_diff_scale_shift, x_tmp_1);
                    ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));
                    add(x_tmp_0, XReg(IDX(reg_ws)), XReg(IDX(reg_coff)));
                    if (coff) add_imm(x_tmp_0, x_tmp_0, coff, x_tmp_1);
                    uni_load_maybe_tail(vdiff_gamma, x_tmp_0);
                    add(x_tmp_0, XReg(IDX(reg_ws)), XReg(IDX(reg_coff)));
                    if (coff || chan_data_offt)
                        add_imm(x_tmp_0, x_tmp_0, coff + chan_data_offt,
                                x_tmp_1);
                    uni_load_maybe_tail(vdiff_beta, x_tmp_0);
                    add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws_off_copy,
                            x_tmp_1);
                    ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));

                    fmul(ZRegS(IDX(vdiff_gamma)), ZRegS(IDX(vdiff_gamma)),
                            ZRegS(IDX(vsqrtvar)));
                    uni_vdivps_aarch64(
                            vdiff_beta, vdiff_beta, vchan_size, p_512);
                    uni_vdivps_aarch64(
                            vdiff_gamma, vdiff_gamma, vchan_size, p_512);

                    add(x_tmp_0, XReg(IDX(reg_diff_dst)),
                            XReg(IDX(reg_soff_nspc)));
                    if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                    uni_load_spat_data(ZReg(idx), x_tmp_0);

                    if (with_relu) {
                        if (isa == sve_512)
                            bwd_process_relu_avx512_common(ZReg(idx), offt);
                        else
                            assert(false);
                    }

                    if (!bdesc_->use_global_stats()) {
                        fsub(ZRegS((uint32_t)idx), ZRegS((uint32_t)idx),
                                ZRegS(IDX(vdiff_beta)));
                        add(x_tmp_0, XReg(IDX(reg_src)),
                                XReg(IDX(reg_soff_nspc)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_load_spat_data(ZReg(idx + 1), x_tmp_0);
                        fsub(ZRegS((uint32_t)idx + 1), ZRegS(IDX(vmean)),
                                ZRegS((uint32_t)idx + 1));
                        fmul(ZRegS((uint32_t)idx + 1), ZRegS((uint32_t)idx + 1),
                                ZRegS(IDX(vdiff_gamma)));
                        uni_vaddps_unpredicate_aarch64(
                                ZReg(idx), ZReg(idx), ZReg(idx + 1));
                    }

                    fmul(ZRegS((uint32_t)idx), ZRegS((uint32_t)idx),
                            ZRegS(IDX(vsqrtvar)));

                    if (bdesc_->use_scaleshift()) {
                        fmul(ZRegS((uint32_t)idx), ZRegS((uint32_t)idx),
                                ZRegS(IDX(vgamma)));
                    }

                    if (stream_store_allowed) {
                        uni_vmovntps_aarch64(reg_diff_src, reg_soff_nspc, offt,
                                ZReg(idx), p_512, p_lsb_256, p_lsb_128);

                    } else {
                        add(x_tmp_0, XReg(IDX(reg_diff_src)),
                                XReg(IDX(reg_soff_nspc)));
                        if (offt) add_imm(x_tmp_0, x_tmp_0, offt, x_tmp_1);
                        uni_store_spat_data(x_tmp_0, ZReg(idx));
                    }

                    coff += vlen;
                    offt += vlen_spat_data_;
                }
                add_imm(XReg(IDX(reg_soff_nspc)), XReg(IDX(reg_soff_nspc)),
                        spat_step, x_tmp_0);
                sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), num_spat_pts,
                        x_tmp_0);
                cbnz(XReg(IDX(reg_ctr)), spatial);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            cmp(XReg(IDX(reg_diff_src)), vlen - 1);
            cbnz(XReg(IDX(reg_diff_src)), normal_store);
            compute(true);
            b(end_store);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // no NT store for BF16
        }
    }

    void backward_diff_channels_nspc() {
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        mov(XReg(IDX(reg_coff_max_bwd_copy)), XReg(IDX(reg_coff_max)));

        Label ch_unroll_label[5];
        const int max_ch_unroll
                //= is_bf16_ && !mayiuse(avx512_core_bf16) ? 1 : 3;
                = is_bf16_ ? 1 : 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(XReg(IDX(reg_coff_max)), vlen * ch_blk_size);
                b(LT, ch_unroll_label[ch_idx - 1]);

                backward_diff_channels_nspc_compute(ch_blk_size);

                add_imm(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);
                if (!bdesc_->use_global_stats())
                    add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)),
                            vlen_spat_data_ * ch_blk_size, x_tmp_0);
                add_imm(XReg(IDX(reg_diff_src)), XReg(IDX(reg_diff_src)),
                        vlen_spat_data_ * ch_blk_size, x_tmp_0);

                // advance mean_ptr() and var_ptr()
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)),
                        vlen * ch_blk_size, x_tmp_0);

                add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), 2 * ch_blk_size,
                        x_tmp_0);

                sub_imm(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)),
                        vlen * ch_blk_size, x_tmp_0);
                b(ch_unroll_label[ch_idx]);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max_bwd_copy)));
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_scale_shift,
                x_tmp_1);
        ldr(XReg(IDX(reg_diff_scale_shift)), ptr(x_tmp_0));

        //if (is_bf16_) shr(reg_coff_max, 1);
        sub(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                XReg(IDX(reg_coff_max)));
        if (!bdesc_->use_global_stats())
            sub(XReg(IDX(reg_src)), XReg(IDX(reg_src)),
                    XReg(IDX(reg_coff_max)));
        sub(XReg(IDX(reg_diff_src)), XReg(IDX(reg_diff_src)),
                XReg(IDX(reg_coff_max)));
        //if (is_bf16_) shl(reg_coff_max, 1);

        lsr(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
        sub(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), XReg(IDX(reg_coff_max)));
        lsl(XReg(IDX(reg_coff_max)), XReg(IDX(reg_coff_max)), 5 % 64);
    }

    void backward() {
        eor(ZRegD(0), ZRegD(0), ZRegD(0));
        eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
        Label zero_rbuf, sh_spatial;

        L(zero_rbuf);
        {
            add(x_tmp_0, XReg(IDX(reg_rbuf1)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));
            add(x_tmp_0, XReg(IDX(reg_rbuf2)), XReg(IDX(reg_coff)));
            uni_vmovups_aarch64(x_tmp_0, ZReg(0));
            add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen, x_tmp_0);
            cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
            b(NE, zero_rbuf);
        }

        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
        ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_dst, x_tmp_1);
        ldr(XReg(IDX(reg_diff_dst)), ptr(x_tmp_0));
        if (with_relu) {
            assert(isa == sve_512);
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws, x_tmp_1);
            ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));
        }

        eor(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), XReg(IDX(reg_soff)));
        L(sh_spatial);
        {
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
            is_nspc_ ? backward_sh_channels_nspc() : backward_sh_channels();
            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                        mb_offt, x_tmp_0);
                add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), mb_offt,
                        x_tmp_0);
                add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), ws_mb_offt,
                        x_tmp_0);
            } else {
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        XReg(IDX(reg_mb_stride_Bc)));
            }
            cmp(XReg(IDX(reg_soff)), XReg(IDX(reg_soff_max)));
            b(LT, sh_spatial);
        }

        if (is_nspc_) {
            // comeback
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
            ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_dst, x_tmp_1);
            ldr(XReg(IDX(reg_diff_dst)), ptr(x_tmp_0));
        }

        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_scale_shift,
                x_tmp_1);
        ldr(XReg(IDX(reg_diff_scale_shift)), ptr(x_tmp_0));

        Label no_sh_reduction;
        barrier();
        {
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_ithr, x_tmp_1);
            ldr(XReg(IDX(reg_tmp)), ptr(x_tmp_0));
            cmp(XReg(IDX(reg_tmp)), 0);
            Label sh_reduction_channels;
            b(NE, no_sh_reduction);

            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_N_nthr, x_tmp_1);
            ldr(XReg(IDX(reg_nnthr)), ptr(x_tmp_0));
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
            L(sh_reduction_channels);
            {
                mov(XReg(IDX(reg_roff)), XReg(IDX(reg_coff)));
                eor(ZRegD(0), ZRegD(0), ZRegD(0));
                eor(ZRegD(1), ZRegD(1), ZRegD(1));
                uni_load_maybe_tail(vsqrtvar, var_ptr());
                uni_vaddps_unpredicate_aarch64(vsqrtvar, vsqrtvar, veps);
                uni_vsqrtps_aarch64(vsqrtvar, vsqrtvar, p_512);
                uni_vdivps_aarch64(vsqrtvar, vone, vsqrtvar, p_512);
                mov(XReg(IDX(reg_ctr)), XReg(IDX(reg_nnthr)));
                Label sh_reduction_thrs;
                L(sh_reduction_thrs);
                { // TODO: unroll (?)
                    XReg x_roff(IDX(reg_roff));

                    add(x_tmp_0, XReg(IDX(reg_rbuf1)), x_roff);
                    add(x_tmp_1, XReg(IDX(reg_rbuf2)), x_roff);
		    if (isa == sve_512) {
                        ld1w(z_tmp0.s, p_512 / T_z, ptr(x_tmp_0));
                        ld1w(z_tmp1.s, p_512 / T_z, ptr(x_tmp_1));
#if 0
                    } else if (ZReg(0).isYMM()) {
                        ld1w(z_tmp0.s, p_lsb_256 / T_z,
                                ptr(x_tmp_0));
                        ld1w(z_tmp1.s, p_lsb_256 / T_z,
                                ptr(x_tmp_1));
#endif // #if 0
                    } else {
                        ld1(VReg4S(tmp_vec_idx[0]), ptr(x_tmp_0));
                        ld1(VReg4S(tmp_vec_idx[1]), ptr(x_tmp_1));
                    }
                    uni_vaddps_unpredicate_aarch64(
                            ZReg(0), ZReg(0), ZReg(tmp_vec_idx[0]));
                    uni_vaddps_unpredicate_aarch64(
                            ZReg(1), ZReg(1), ZReg(tmp_vec_idx[1]));
                    add(XReg(IDX(reg_roff)), XReg(IDX(reg_roff)),
                            XReg(IDX(reg_coff_max)));
                    sub_imm(XReg(IDX(reg_ctr)), XReg(IDX(reg_ctr)), 1, x_tmp_0);
                    cbnz(XReg(IDX(reg_ctr)), sh_reduction_thrs);
                }
                fmul(ZRegS(0), ZRegS(0), ZRegS(IDX(vsqrtvar)));
                uni_store_maybe_tail(diff_gamma_ptr(), ZReg(0));
                uni_store_maybe_tail(diff_beta_ptr(), ZReg(1));
                add_imm(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), vlen,
                        x_tmp_0);
                cmp(XReg(IDX(reg_coff)), XReg(IDX(reg_coff_max)));
                b(NE, sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_src, x_tmp_1);
        ldr(XReg(IDX(reg_diff_src)), ptr(x_tmp_0));
        if (with_relu) {
            assert(isa == sve_512);
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws, x_tmp_1);
            ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));
        }

        eor(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), XReg(IDX(reg_soff)));
        Label diff_spatial;
        L(diff_spatial);
        {
            eor(XReg(IDX(reg_coff)), XReg(IDX(reg_coff)), XReg(IDX(reg_coff)));
            is_nspc_ ? backward_diff_channels_nspc() : backward_diff_channels();
            // Process next image
            if (is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (!bdesc_->use_global_stats() && mb_offt)
                    add_imm(XReg(IDX(reg_src)), XReg(IDX(reg_src)), mb_offt,
                            x_tmp_0);
                if (mb_offt)
                    add_imm(XReg(IDX(reg_diff_dst)), XReg(IDX(reg_diff_dst)),
                            mb_offt, x_tmp_0);
                if (mb_offt)
                    add_imm(XReg(IDX(reg_diff_src)), XReg(IDX(reg_diff_src)),
                            mb_offt, x_tmp_0);
                if (mb_offt)
                    add_imm(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)), mb_offt,
                            x_tmp_0);
                if (ws_mb_offt)
                    add_imm(XReg(IDX(reg_ws)), XReg(IDX(reg_ws)), ws_mb_offt,
                            x_tmp_0);
            } else {
                add(XReg(IDX(reg_soff)), XReg(IDX(reg_soff)),
                        XReg(IDX(reg_mb_stride_Bc)));
            }
            cmp(XReg(IDX(reg_soff)), XReg(IDX(reg_soff_max)));
            b(LT, diff_spatial);
        }
        if (is_nspc_) {
            // comeback
            if (!bdesc_->use_global_stats()) {
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_src, x_tmp_1);
                ldr(XReg(IDX(reg_src)), ptr(x_tmp_0));
            }
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_dst, x_tmp_1);
            ldr(XReg(IDX(reg_diff_dst)), ptr(x_tmp_0));
            add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_diff_src, x_tmp_1);
            ldr(XReg(IDX(reg_diff_src)), ptr(x_tmp_0));
            if (with_relu) {
                add_imm(x_tmp_0, XReg(IDX(rsp)), (int)stack_off_ws, x_tmp_1);
                ldr(XReg(IDX(reg_ws)), ptr(x_tmp_0));
            }
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *bdesc) : bdesc_(bdesc) {
        static_assert(isa == asimd || isa == sve_512, "unsupported isa");

        const int simd_w = isa == asimd
                ? 8
                : cpu_isa_traits<isa>::vlen / sizeof(acc_data_t);
        is_bf16_ = bdesc_->desc()->data_desc.data_type == data_type::bf16;
        size_t dt_size
                = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);
        is_spatial_thr_ = bnorm_utils::is_spatial_thr(
                bdesc_, is_nspc_, simd_w, dt_size);
        vlen_spat_data_ = vlen / (1 + is_bf16_); // 32B of BF16 -> 64B of FP32

        unroll_blocks = isa == sve_512 && !is_spatial_thr_ ? 4 : 1;
        unroll_regs = isa == sve_512 && !is_spatial_thr_ ? 4 : 1;
    }

    void generate() override {
        preamble();

#if 0
        if (is_bf16_) {
            // init emulation of bfloat16 operations
            if (!mayiuse(avx512_core_bf16)) {
                bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserved_1,
                        bf16_emu_reserved_2, bf16_emu_reserved_3, reg_bf16_tmp,
                        bf16_emu_reserved_4, bf16_emu_reserved_4);
                bf16_emu_->init_vcvtneps2bf16();
            }
        }
#endif // #if 0

        ptrue(p_512.b);
        ptrue(p_lsb_256.b, VL32);
        ptrue(p_lsb_128.b, VL16);

        if (isa == sve_512) prepare_tail_mask_avx512_common();

        compute_static_strides();
        sub_imm(XReg(IDX(rsp)), XReg(IDX(rsp)), (int)stack_size_required,
                x_tmp_0);
        load_common_params();
        prepare_relu();

        if (bdesc_->is_fwd()) {
            if (!bdesc_->stats_is_src()) { compute_mean_variance(); }
            forward();
        } else {
            backward();
        }
        add_imm(XReg(IDX(rsp)), XReg(IDX(rsp)), (int)stack_size_required,
                x_tmp_0);
        postamble();
    }

    void operator()(const call_params_t *p) { jit_generator::operator()(p); }
};
} // namespace

namespace bnorm_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *bdesc)
        : bdesc_(bdesc), ker_(bdesc_) {
        const dim_t C_PADDED = get_c_padded(bdesc_);

        const memory_desc_wrapper src_d(bdesc_->src_md());
        is_nspc_
                = src_d.matches_one_of_tag(format_tag::nhwc, format_tag::ndhwc);

        dt_size_ = types::data_type_size(bdesc_->desc()->data_desc.data_type);
        size_t data_size = dt_size_ * bdesc_->MB() * C_PADDED * bdesc_->D()
                * bdesc_->H() * bdesc_->W();
        l3_size_ = platform::get_per_core_cache_size(3) * dnnl_get_max_threads()
                / 2; // XXX
        // TODO: cache balancing for nspc
        do_blocking_ = is_nspc_ ? false
                                : (data_size >= l3_size_ / 2 && l3_size_ > 0);
    }

    ~driver_t() = default;

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *bdesc) {
        dim_t C_PADDED = get_c_padded(bdesc);

        int sbuf_sz = use_tmp_stats(bdesc) * 2 * C_PADDED;
        int pbuf_sz = use_tmp_diff_scale_shift(bdesc) * 2 * C_PADDED;
        int rbuf_sz
                = (bdesc->is_fwd() ? 1 : 2) * C_PADDED * dnnl_get_max_threads();

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);

        if (dnnl_thr_syncable()) {
            int n_barriers = C_PADDED / simd_w;
            scratchpad.book<barrier::ctx_64_t>(key_barrier, n_barriers);
        }
    }

    void exec(int ithr, int nthr, const void *src, void *diff_src, void *dst,
            const void *diff_dst, const acc_data_t *scale_shift,
            acc_data_t *diff_scale_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
        auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);

        dim_t N = bdesc_->MB();
        dim_t C = bdesc_->C();
        dim_t C_PADDED = get_c_padded(bdesc_);
        dim_t D = bdesc_->D();
        dim_t H = bdesc_->H();
        dim_t W = bdesc_->W();
        dim_t SP = D * H * W;
        dim_t img_size = C_PADDED * D * H * W;
        const int vlen_spat_data = ker_.spat_step;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = bdesc_->desc()->batch_norm_epsilon;
        p.one = 1.0f;
        p.spat_size = D * H * W;
        p.chan_size = 1.0f * N * p.spat_size;

        dim_t C_blks = C_PADDED / simd_w;

        int C_ithr {0}, C_nthr {0}, N_ithr {0}, N_nthr {0}, S_ithr {0},
                S_nthr {0};
        dim_t C_blk_s {0}, C_blk_e {0}, N_s {0}, N_e {0}, S_s {0}, S_e {0};

        dim_t C_blks_per_iter {1};
        int64_t iters {1};
        if (do_blocking_) {
            int num_tensors = bdesc_->is_fwd() ? 1 : 2;
            size_t working_set_size
                    = dt_size_ * (N * D * H * W * simd_w) * num_tensors;
            bnorm_utils::cache_balance(
                    working_set_size, C_blks, N, nthr, C_blks_per_iter, iters);
        }

        bool spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                true /* spatial_thr_allowed */, is_nspc_, ithr, nthr, N,
                do_blocking_ ? C_blks_per_iter : C_blks, SP,
                /* outputs */ C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr,
                N_s, N_e, S_ithr, S_nthr, S_s, S_e);

        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;
        assert(IMPLICATION(!dnnl_thr_syncable(), SP_N_nthr == 1));

        p.N_ithr = SP_N_ithr;
        p.N_nthr = SP_N_nthr;

        int last_iter_blks = C_blks - (iters - 1) * C_blks_per_iter;
        int global_C_blk_s;
        int global_barriers_per_iter = C_nthr;

        for (int64_t it = 0; it < iters; it++) {
            if (it == iters - 1 && iters > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking_,
                        spatial_thr_allowed, is_nspc_, ithr, nthr, N,
                        last_iter_blks, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);

                // Update call parameters for JIT, last iteration
                p.N_ithr = N_ithr * S_nthr + S_ithr;
                p.N_nthr = N_nthr * S_nthr;
            }

            global_C_blk_s = do_blocking_
                    ? (C_blk_s == -1) ? -1 : it * C_blks_per_iter + C_blk_s
                    : C_blk_s;

            int C_blks_thr = C_blk_e - C_blk_s;
            int N_thr = N_e - N_s;

            size_t coff_base = global_C_blk_s * simd_w;
            size_t soff_base = is_nspc_
                    ? coff_base + N_s * img_size
                    : global_C_blk_s * p.spat_size * simd_w + N_s * img_size;

            p.spat_size_loc = S_e - S_s;
            p.S_s = S_s * vlen_spat_data;
            p.S_tail = (p.spat_size - S_e) * vlen_spat_data;
            p.coff_max = C_blks_thr * simd_w;
            p.mean = (use_tmp_stats(bdesc_) ? sbuf : mean) + coff_base;
            p.var = (use_tmp_stats(bdesc_) ? sbuf + C_PADDED : var) + coff_base;
            p.scale_shift = scale_shift + coff_base;
            p.diff_scale_shift
                    = (use_tmp_diff_scale_shift(bdesc_) ? pbuf
                                                        : diff_scale_shift)
                    + coff_base;

            p.soff_max = dt_size_ * N_thr * img_size;
            p.src = (void *)((char *)src + soff_base * dt_size_);
            p.dst = (void *)((char *)dst + soff_base * dt_size_);
            p.diff_src = (void *)((char *)diff_src + soff_base * dt_size_);
            p.diff_dst = (void *)((char *)diff_dst + soff_base * dt_size_);
            p.ws = ws + soff_base / 8;

            p.mb_stride_Bc = dt_size_ * (img_size - p.coff_max * p.spat_size);

            // use SP_N_nthr which is the same as p.N_nthr except maybe for
            // the last iteration.
            p.rbuf1 = rbuf
                    + ((it * C_blks_per_iter) * SP_N_nthr + C_blk_s * p.N_nthr
                              + p.N_ithr * C_blks_thr)
                            * simd_w;
            // rbuf1 and rbuf2 have to be disjoint
            p.rbuf2 = p.rbuf1 + C_PADDED * nthr;
            p.is_cblk_tail = (it * C_blks_per_iter + C_blk_e) * simd_w > C;

            size_t iter_bariers
                    = do_blocking_ ? it * global_barriers_per_iter : 0;
            p.barrier = barriers + C_ithr + iter_bariers;
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    void init_barriers(const memory_tracking::grantor_t &scratchpad) {
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);
        if (barriers) {
            const int n_barriers = get_c_padded(bdesc_) / simd_w;
            for (int i = 0; i < n_barriers; ++i)
                barrier::ctx_init(&barriers[i]);
        }
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    enum {
        simd_w = isa == asimd ? 8
                              : cpu_isa_traits<isa>::vlen
                        / sizeof(acc_data_t) // BF16 will expand to FP32
    };

    static bool use_tmp_stats(const batch_normalization_pd_t *bdesc) {
        return true && !bdesc->stats_is_src()
                && bdesc->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale_shift(
            const batch_normalization_pd_t *bdesc) {
        return false || (bdesc->is_bwd() && !bdesc->use_scaleshift())
                || bdesc->desc()->prop_kind == prop_kind::backward_data;
    }

    static dim_t get_c_padded(const batch_normalization_pd_t *bdesc) {
        return bdesc->src_md()->padded_dims[1];
    }

    const batch_normalization_pd_t *bdesc_;
    jit_bnorm_t<isa> ker_;
    bool do_blocking_;
    bool is_nspc_;
    size_t l3_size_;
    size_t dt_size_;
};
} // namespace bnorm_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_fwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && one_of(src_md()->data_type, f32, bf16)
            && IMPLICATION(src_md()->data_type == bf16, false)
            && check_scale_shift_data_type()
            && (attr()->has_default_values() || this->with_relu_post_op());
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    if (isa == sve_512) {
        if (!src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc))
            return status::unimplemented;
    } else {
        if (!src_d.matches_one_of_tag(nChw8c, nCdhw8c))
            return status::unimplemented;
    }

    if (is_training() && fuse_norm_relu()) {
        if (isa < sve_512) return status::unimplemented;
        init_default_ws(1);
    }

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < sve_512)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale_shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);

    auto mean = pd()->stats_is_src() ? const_cast<acc_data_t *>(
                        CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN))
                                     : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
    auto var = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                    CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, nullptr, dst, nullptr, scale_shift,
                nullptr, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = true
            /* the algorithm requires barriers for best performance so for TBB we use
             * jit_uni_tbb_batch_normalization instead */
            && dnnl_thr_syncable() && mayiuse(isa) && is_bwd()
            && !has_zero_dim_memory() && one_of(ndims(), 4, 5)
            && set_default_formats_common()
            && one_of(true,
                    everyone_is(
                            f32, src_md()->data_type, diff_src_md()->data_type),
                    everyone_is(bf16, src_md()->data_type,
                            diff_src_md()->data_type))
            && IMPLICATION(src_md()->data_type == bf16, false)
            && check_scale_shift_data_type() && attr()->has_default_values();
    if (!ok) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());

    format_tag_t src_tag, diff_src_tag;
    if (isa == sve_512) {
        src_tag = src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc);
        diff_src_tag
                = diff_src_d.matches_one_of_tag(nChw16c, nCdhw16c, nhwc, ndhwc);
    } else {
        src_tag = src_d.matches_one_of_tag(nChw8c, nCdhw8c);
        diff_src_tag = diff_src_d.matches_one_of_tag(nChw8c, nCdhw8c);
    }
    ok = (src_tag != format_tag::undef && diff_src_tag != format_tag::undef
            && src_tag == diff_src_tag);
    if (!ok) return status::unimplemented;

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C() && isa < sve_512)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    if (fuse_norm_relu()) {
        if (isa < sve_512) return status::unimplemented;
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    /* TODO: extra checks required */

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_, new bnorm_impl::driver_t<isa>(pd())));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto var = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale_shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    auto diff_scale_shift
            = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->init_barriers(scratchpad);

    parallel(0, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, diff_src, nullptr, diff_dst,
                scale_shift, diff_scale_shift, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<asimd>;
template struct jit_uni_batch_normalization_bwd_t<asimd>;
template struct jit_uni_batch_normalization_fwd_t<sve_512>;
template struct jit_uni_batch_normalization_bwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
