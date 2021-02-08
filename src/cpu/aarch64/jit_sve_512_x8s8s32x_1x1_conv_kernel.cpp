/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/aarch64/jit_sve_512_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/aarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof(jit_1x1_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;

#define SVE_compress_addr(base, offt) \
    Xbyak_aarch64::ptr(get_comp_addr_reg(base, offt))

template <typename Vmm>
void _jit_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::bcast_loop(int load_loop_blk) {
    xa_->mov(aux1_reg_bcast_data, reg_bcast_data);
    xa_->mov(aux_reg_bcast_data, reg_bcast_data);

    xa_->mov(aux_reg_output_data, reg_output_data);
    ldr(bcast_loop_iter, SVE_compress_addr(reg_rsp, bcast_loop_work_off));

    Label bcast_loop;
    Label bcast_loop_tail;

    xa_->cmp_imm(bcast_loop_iter, jcp.ur, reg_tmp0_imm);
    b(LT, bcast_loop_tail);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_substep, reg_tmp0_imm);
                add_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_substep, reg_tmp0_imm);
            } else {
                add_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp0_imm);
                int output_offset = jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep;

                add_imm(aux_reg_output_data, aux_reg_output_data, output_offset,
                        reg_tmp0_imm);
            }
        }
        subs(bcast_loop_iter, bcast_loop_iter, jcp.bcast_block);
        xa_->cmp_imm(bcast_loop_iter, jcp.bcast_block, reg_tmp0_imm);
        b(GE, bcast_loop);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        xa_->cmp_imm(bcast_loop_iter, 0, reg_tmp0_imm);
        b(EQ, bcast_loop_tail_out);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

template <typename Vmm>
void _jit_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {
    auto vreg_load
            = [=](int i_load) { return ZReg(ur * load_loop_blk + i_load); };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return ZReg(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](ZReg bias_reg, int i_load, bool mask_flag) {
        int offt = get_offset(jcp.typesize_bia * jcp.oc_block * i_load);

        add_imm(reg_tmp0_adr, reg_bias_data, offt, reg_tmp0_imm);
        if (mask_flag)
            ld1w(bias_reg.s, ktail_mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(reg_tmp0_adr));
        else
            ldr(bias_reg, Xbyak_aarch64::ptr(reg_tmp0_adr));
    };

    auto bias_ptr8 = [=](ZReg bias_reg, int i_load, bool mask_flag) {
        int offt = get_offset(jcp.typesize_bia * jcp.oc_block * i_load);

        add_imm(reg_tmp0_adr, reg_bias_data, offt, reg_tmp0_imm);
        if (mask_flag) {
            uzp1(ktail_load_mask.h, ktail_mask.h, mask_all_zero.h);
            uzp1(ktail_load_mask.b, ktail_load_mask.b, mask_all_zero.b);
            ld1b(bias_reg.b, ktail_load_mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(reg_tmp0_adr));
        } else {
            ldr(QReg(bias_reg.getIdx()), Xbyak_aarch64::ptr(reg_tmp0_adr));
        }
    };

    auto comp_ptr = [=](ZReg comp_reg, int i_load, bool mask_flag) {
        int offt = get_offset(sizeof(int32_t) * jcp.oc_block * i_load);

        add_imm(reg_tmp0_adr, reg_comp_data, offt, reg_tmp0_imm);
        if (mask_flag)
            ld1w(comp_reg.s, ktail_mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(reg_tmp0_adr));
        else
            ldr(comp_reg, Xbyak_aarch64::ptr(reg_tmp0_adr));
    };

    auto scale_ptr = [=](ZReg scale_reg, int i_load) {
        int ofs = get_offset(
                jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load));

        if (ofs == 0) {
            ldr(scale_reg, Xbyak_aarch64::ptr(reg_ptr_scales));
        } else {
            auto reg_tmp_adr = ((i_load % 4) == 0) ? reg_tmp0_adr
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_adr
                            : ((i_load % 4) == 2) ? reg_tmp2_adr : reg_tmp3_adr;
            auto reg_tmp_imm = ((i_load % 4) == 0) ? reg_tmp0_imm
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_imm
                            : ((i_load % 4) == 2) ? reg_tmp2_imm : reg_tmp3_imm;
            add_imm(reg_tmp_adr, reg_ptr_scales, ofs, reg_tmp_imm);
            ldr(scale_reg, Xbyak_aarch64::ptr(reg_tmp_adr));
        }
    };

    auto bcast_ptr = [=](ZReg bcast_reg, int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int _offt = (jcp.ic_without_padding * i_ur * jcp.ngroups + i_reduce);

        //        return EVEX_compress_addr(
        //                aux_reg_bcast_data, jcp.typesize_in * offt, bcast);

        auto base = aux_reg_bcast_data;
        auto ofs = get_offset(jcp.typesize_in * _offt);

        if (bcast)
            assert(!"unimplemented");
        else {
            if ((-0x40 <= ofs) && (ofs < 0x40) && ((ofs % 4) == 0))
                ld1rw(ZRegS(bcast_reg.getIdx()), PReg(vmask.getIdx()),
                        Xbyak_aarch64::ptr(XReg(base.getIdx()),
                                static_cast<int32_t>(ofs)));
            else {
                auto reg_tmp_adr = ((i_ur % 4) == 0)
                        ? reg_tmp0_adr
                        : ((i_ur % 4) == 1) ? reg_tmp1_adr
                                            : ((i_ur % 4) == 2) ? reg_tmp2_adr
                                                                : reg_tmp3_adr;
                auto reg_tmp_imm = ((i_ur % 4) == 0)
                        ? reg_tmp0_imm
                        : ((i_ur % 4) == 1) ? reg_tmp1_imm
                                            : ((i_ur % 4) == 2) ? reg_tmp2_imm
                                                                : reg_tmp3_imm;
                add_imm(reg_tmp_adr, XReg(base.getIdx()), ofs, reg_tmp_imm);
                ld1rw(ZRegS(bcast_reg.getIdx()), PReg(vmask.getIdx()),
                        Xbyak_aarch64::ptr(reg_tmp_adr));
            }
        }
    };

    auto load_ptr = [=](ZReg load_reg, int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        int ofs = get_offset(
                u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt);

        if (ofs == 0) {
            ldr(load_reg, Xbyak_aarch64::ptr(aux_reg_load_data));
        } else {
            auto reg_tmp_adr = ((i_load % 4) == 0) ? reg_tmp0_adr
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_adr
                            : ((i_load % 4) == 2) ? reg_tmp2_adr : reg_tmp3_adr;
            auto reg_tmp_imm = ((i_load % 4) == 0) ? reg_tmp0_imm
                                                   : ((i_load % 4) == 1)
                            ? reg_tmp1_imm
                            : ((i_load % 4) == 2) ? reg_tmp2_imm : reg_tmp3_imm;
            add_imm(reg_tmp_adr, aux_reg_load_data, ofs, reg_tmp_imm);
            ldr(load_reg, Xbyak_aarch64::ptr(reg_tmp_adr));
        }
    };

    auto output_ptr = [=](ZReg output_reg, int i_load, int i_ur,
                              bool mask_flag) {
        const size_t ur_stride = jcp.with_dw_conv
                ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                : jcp.oc_without_padding * jcp.ngroups * i_ur;

        int offt = get_offset(
                jcp.typesize_out * (ur_stride + i_load * jcp.load_block));

        add_imm(reg_tmp0_adr, aux_reg_output_data, offt, reg_tmp0_imm);
        if (mask_flag)
            ld1w(output_reg.s, ktail_mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(reg_tmp0_adr));
        else
            ldr(output_reg, Xbyak_aarch64::ptr(reg_tmp0_adr));
    };

    auto output_ptr8 = [=](ZReg output_reg, int i_load, int i_ur,
                               bool mask_flag) {
        const size_t ur_stride = jcp.with_dw_conv
                ? jcp.nb_load_blocking * jcp.oc_block * i_ur
                : jcp.oc_without_padding * jcp.ngroups * i_ur;

        int offt = get_offset(
                jcp.typesize_out * (ur_stride + i_load * jcp.load_block));

        add_imm(reg_tmp0_adr, aux_reg_output_data, offt, reg_tmp0_imm);
        if (mask_flag) {
            uzp1(ktail_load_mask.h, ktail_mask.h, mask_all_zero.h);
            uzp1(ktail_load_mask.b, ktail_load_mask.b, mask_all_zero.b);
            ld1b(output_reg.b, ktail_load_mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(reg_tmp0_adr));
        } else {
            ldr(QReg(output_reg.getIdx()), Xbyak_aarch64::ptr(reg_tmp0_adr));
        }
    };

    auto init = [=]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                eor(r.d, r.d, r.d);
            }
        if (!jcp.signed_input) {
            //xa_->mov(reg_scratch, -128);
            //vpbroadcastb(vmm_shift, reg_scratch.cvt8());
            dup(vmm_shift.b, -128);
        }
    };

    auto store = [=](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = nullptr;
        if (sum_idx != -1) p_sum_scale = &p.entry_[sum_idx].sum.scale;
        str(reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        ldr(reg_ptr_scales, SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        if (p_sum_scale && *p_sum_scale != 1.f) {
            str(reg_load_data, SVE_compress_addr(reg_rsp, reg_load_data_off));
            xa_->mov_imm(reg_ptr_sum_scale, (size_t)p_sum_scale);
        }
#if 0
        if (jcp.src_zero_point) {
            ldr(reg_zp_compensation,
                    SVE_compress_addr(reg_rsp, reg_zp_compensation_off));
            ldr(reg_src_zero_point,
                    SVE_compress_addr(reg_rsp, reg_src_zero_point_off));
        }
#endif
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto vmm_bias = vmm_tmp;
            auto vmm_comp = vmm_bcast;
            if (jcp.with_bias) {
                if (!jcp.signed_input)
                    ldr(reg_bias_data,
                            SVE_compress_addr(reg_rsp, reg_bias_data_off));
                //cvt2ps(jcp.bia_dt, vmm_bias, bias_ptr(i_load), mask_flag);
                switch (jcp.bia_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        bias_ptr(vmm_bias, i_load, mask_flag);
                        break;
#if 0
		    case data_type::s8:
                        xa_->sub(x22, x22, 64);
                        str(ZReg(29), Xbyak_aarch64::ptr(x22));
                        bias_ptr8(ZReg(29), i_load, mask_flag);
                        zip1(ZRegB(29), ZRegB(29), ZRegB(29));
                        zip1(ZRegH(29), ZRegH(29), ZRegH(29));
                        sxtb(ZRegS(vmm_bias.getIdx()),
                                vmask / Xbyak_aarch64::T_m, ZRegS(29));
                        if (mask_flag) {
                            xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            xa_->mov(vmm_bias.s, mask_tmp / Xbyak_aarch64::T_m,
                                    0);
                        }
                        ldr(ZReg(29), Xbyak_aarch64::ptr(x22));
                        xa_->add(x22, x22, 64);
                        break;
                    case data_type::u8:
                        xa_->sub(x22, x22, 64);
                        str(ZReg(29), Xbyak_aarch64::ptr(x22));
                        bias_ptr8(ZReg(29), i_load, mask_flag);
                        zip1(ZRegB(29), ZRegB(29), ZRegB(29));
                        zip1(ZRegH(29), ZRegH(29), ZRegH(29));
                        uxtb(ZRegS(vmm_bias.getIdx()),
                                vmask / Xbyak_aarch64::T_m, ZRegS(29));
                        if (mask_flag) {
                            xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            xa_->mov(vmm_bias.s, mask_tmp / Xbyak_aarch64::T_m,
                                    0);
                        }
                        ldr(ZReg(29), Xbyak_aarch64::ptr(x22));
                        xa_->add(x22, x22, 64);
                        break;
#else
                    case data_type::s8:
                        bias_ptr8(ZReg(17), i_load, mask_flag);
                        zip1(ZRegB(17), ZRegB(17), ZRegB(17));
                        zip1(ZRegH(17), ZRegH(17), ZRegH(17));
                        sxtb(ZRegS(vmm_bias.getIdx()),
                                vmask / Xbyak_aarch64::T_m, ZRegS(17));
                        if (mask_flag) {
                            xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            xa_->mov(vmm_bias.s, mask_tmp / Xbyak_aarch64::T_m,
                                    0);
                        }
                        break;
                    case data_type::u8:
                        bias_ptr8(ZReg(17), i_load, mask_flag);
                        zip1(ZRegB(17), ZRegB(17), ZRegB(17));
                        zip1(ZRegH(17), ZRegH(17), ZRegH(17));
                        uxtb(ZRegS(vmm_bias.getIdx()),
                                vmask / Xbyak_aarch64::T_m, ZRegS(17));
                        if (mask_flag) {
                            xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                            xa_->mov(vmm_bias.s, mask_tmp / Xbyak_aarch64::T_m,
                                    0);
                        }
                        break;
#endif
                    default: assert(!"unsupported data type");
                }
                if (jcp.bia_dt != data_type::f32)
                    scvtf(ZRegS(vmm_bias.getIdx()), vmask,
                            ZRegS(vmm_bias.getIdx()));
            }
            if (!jcp.signed_input) {
                ldr(reg_comp_data,
                        SVE_compress_addr(reg_rsp, reg_comp_data_off));
                // cvt2ps(data_type::s32, vmm_comp, comp_ptr(i_load), mask_flag);
                comp_ptr(vmm_comp, i_load, mask_flag);
                scvtf(vmm_comp.s, vmask, vmm_comp.s);
            }
#if 0
            if (jcp.src_zero_point) {
                // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
                const int zp_offset = sizeof(int32_t) * i_load * jcp.load_block;
                // vmovups(vmm_zp,
                //         EVEX_compress_addr(reg_zp_compensation, zp_offset));
                auto reg_addr = get_comp_addr_reg(reg_zp_compensation, zp_offset);
                ldr(vmm_zp.s, Xbyak_aarch64::ptr(reg_addr));

                // vpmulld(vmm_zp, vmm_zp,
                //         EVEX_compress_addr(
                //                 reg_src_zero_point, 0, jcp.zp_src_is_common));
                auto reg_addr2 = get_comp_addr_reg(reg_src_zero_point, 0);
                ldr(vmm_zp2.s, Xbyak_aarch64::ptr(reg_addr));
                mul(vmm_zp.s, vmask, vmm_zp2.s);

                // upscale to f32
                auto vmm_ = mask_flag ? vmm_zp | ktail_mask | Xbyak_aarch64::T_z : vmm_zp;
                // vcvtdq2ps(vmm_, vmm_);
                scvtf(ZRegS(vmm_.getIdx()), vmask, ZRegS(vmm_.getIdx()));
            }
#endif
            auto vmm_scale = vmm_one;
            scale_ptr(vmm_scale, i_load);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                scvtf(ZRegS(r.getIdx()), PReg(vmask.getIdx()),
                        ZRegS(r.getIdx())); //< vcvtdq2ps(r, r);
                if (!jcp.signed_input) xa_->fsub(r.s, r.s, vmm_comp.s);
#if 0
                if (jcp.src_zero_point) xa_->fadd(r.s, r.s, vmm_zp.s);
#endif
                if (jcp.with_bias) xa_->fadd(r.s, r.s, vmm_bias.s);

                // const Vmm mask_vmm = mask_flag ? r | ktail_mask | Xbyak_aarch64::T_z : r;
                zmm_t mask_vmm = r;
                // vmulps(mask_vmm, r, scale_ptr(i_load));
                xa_->fmul(ZRegS(mask_vmm.getIdx()), ZRegS(r.getIdx()),
                        ZRegS(vmm_scale.getIdx()));
                if (mask_flag) {
                    xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                    xa_->mov(ZRegS(mask_vmm.getIdx()),
                            mask_tmp / Xbyak_aarch64::T_m, 0);
                }
            }
        }

        if (p_sum_scale) { // post_op: sum
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
                    auto vmm_prev_dst = vmm_zero;

                    auto r = vreg_accum(i_load, i_ur);
                    // cvt2ps(jcp.dst_dt, vmm_prev_dst, output_ptr(i_load, i_ur),
                    //         mask_flag);
                    switch (jcp.dst_dt) {
                        case data_type::f32:
                        case data_type::s32:
                            output_ptr(vmm_prev_dst, i_load, i_ur, mask_flag);
                            break;
#if 0
			case data_type::s8:
                            xa_->sub(x22, x22, 64);
                            str(ZReg(29), Xbyak_aarch64::ptr(x22));
                            output_ptr8(ZReg(29), i_load, i_ur, mask_flag);
                            zip1(ZRegB(29), ZRegB(29), ZRegB(29));
                            zip1(ZRegH(29), ZRegH(29), ZRegH(29));
                            sxtb(ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / Xbyak_aarch64::T_m, ZRegS(29));
                            if (mask_flag) {
                                xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                xa_->mov(vmm_prev_dst.s,
                                        mask_tmp / Xbyak_aarch64::T_m, 0);
                            }
                            ldr(ZReg(29), Xbyak_aarch64::ptr(x22));
                            xa_->add(x22, x22, 64);
                            break;
                        case data_type::u8:
                            xa_->sub(x22, x22, 64);
                            str(ZReg(29), Xbyak_aarch64::ptr(x22));
                            output_ptr8(ZReg(29), i_load, i_ur, mask_flag);
                            zip1(ZRegB(29), ZRegB(29), ZRegB(29));
                            zip1(ZRegH(29), ZRegH(29), ZRegH(29));
                            uxtb(ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / Xbyak_aarch64::T_m, ZRegS(29));
                            if (mask_flag) {
                                xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                xa_->mov(vmm_prev_dst.s,
                                        mask_tmp / Xbyak_aarch64::T_m, 0);
                            }
                            ldr(ZReg(29), Xbyak_aarch64::ptr(x22));
                            xa_->add(x22, x22, 64);
                            break;
#else
                        case data_type::s8:
                            output_ptr8(ZReg(17), i_load, i_ur, mask_flag);
                            zip1(ZRegB(17), ZRegB(17), ZRegB(17));
                            zip1(ZRegH(17), ZRegH(17), ZRegH(17));
                            sxtb(ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / Xbyak_aarch64::T_m, ZRegS(17));
                            if (mask_flag) {
                                xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                xa_->mov(vmm_prev_dst.s,
                                        mask_tmp / Xbyak_aarch64::T_m, 0);
                            }
                            break;
                        case data_type::u8:
                            output_ptr8(ZReg(17), i_load, i_ur, mask_flag);
                            zip1(ZRegB(17), ZRegB(17), ZRegB(17));
                            zip1(ZRegH(17), ZRegH(17), ZRegH(18));
                            uxtb(ZRegS(vmm_prev_dst.getIdx()),
                                    vmask / Xbyak_aarch64::T_m, ZRegS(17));
                            if (mask_flag) {
                                xa_->not_(mask_tmp.b, vmask.b, ktail_mask.b);
                                xa_->mov(vmm_prev_dst.s,
                                        mask_tmp / Xbyak_aarch64::T_m, 0);
                            }
                            break;
#endif
                        default: assert(!"unsupported data type");
                    }
                    if (jcp.dst_dt != data_type::f32)
                        scvtf(ZRegS(vmm_prev_dst.getIdx()), vmask,
                                ZRegS(vmm_prev_dst.getIdx()));

                    if (*p_sum_scale == 1.f) {
                        // vaddps(r, vmm_prev_dst);
                        xa_->fadd(r.s, r.s, vmm_prev_dst.s);
                    } else {
                        // vfmadd231ps(
                        //         r, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
#if 0
			xa_->sub(x22, x22, 64);
                        str(ZReg(29), Xbyak_aarch64::ptr(x22));
                        ld1rw(ZRegS(29), vmask / Xbyak_aarch64::T_z,
                                Xbyak_aarch64::ptr(reg_ptr_sum_scale));
                        fmla(r.s, vmask / Xbyak_aarch64::T_m, vmm_prev_dst.s,
                                ZRegS(29));
                        ldr(ZReg(29), Xbyak_aarch64::ptr(x22));
                        xa_->add(x22, x22, 64);
#else
                        ld1rw(ZRegS(17), vmask / Xbyak_aarch64::T_z,
                                Xbyak_aarch64::ptr(reg_ptr_sum_scale));
                        fmla(r.s, vmask / Xbyak_aarch64::T_m, vmm_prev_dst.s,
                                ZRegS(17));
#endif
                    }
                }
            }
        }

#if 0
        if (jcp.dst_zero_point) {
//            xa_->mov(reg_dst_zero_point,
//                    EVEX_compress_addr(rsp, reg_dst_zero_point_off));
            ldr(reg_dst_zero_point,
                    SVE_compress_addr(reg_rsp, reg_dst_zero_point_off));

//            vcvtdq2ps(vmm_zp, EVEX_compress_addr(reg_dst_zero_point, 0, true));
            auto reg_addr = get_comp_addr_reg(reg_dst_zero_point, 0);
            ldr(vmm_zp2.s, Xbyak_aarch64::ptr(reg_addr));
            scvtf(ZRegS(vmm_zp.getIdx()), vmask, ZRegS(vmm_zp2.getIdx()));

            /* Add dst zero_point to accumulator */
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(i_load, i_ur);
                    xa_->fadd(r.s, r.s, vmm_zp.s);
                }
            }
        }
#endif

        // Properly saturate the accumulators for integer datatypes
        if (one_of(jcp.dst_dt, u8, data_type::s8, s32)) {
            //            init_saturate_f32(vmm_zero, vmm_saturation,
            //                    reg_ptr_saturation_ubound, f32, jcp.dst_dt);
            //
            if (jcp.dst_dt == data_type::u8) {
                eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
            }
            float saturation_ubound = types::max_value<float>(jcp.dst_dt);
            xa_->mov_imm(reg_tmp0_imm, float2int(saturation_ubound));
            dup(vmm_saturation.s, WReg(reg_tmp0_imm.getIdx()));

#if 0 // optimize instruction order	    
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(i_load, i_ur);
                    // saturate_f32(r, vmm_zero, vmm_saturation, jcp.dst_dt);

                    if (jcp.dst_dt == data_type::u8) {
                        // vmaxps(vmm, vmm, vmm_lbound);
                        fmaxnm(r.s, vmask, vmm_zero.s);
                        fmax(r.s, vmask, vmm_zero.s);
                    }
                    // vminps(vmm, vmm, vmm_ubound);
                    fminnm(r.s, vmask, vmm_saturation.s);
                    fmin(r.s, vmask, vmm_saturation.s);

                    // vcvtps2dq(r, r);
#if 1
                    frintn(r.s, vmask, r.s); // T_rn_sae
                    fcvtzs(r.s, vmask, r.s);
#else
                    frintm(r.s, vmask, r.s); // T_rd_sae
                    fcvtzs(r.s, vmask, r.s);
#endif
                }
            }
#else
            using f = void (CodeGenerator::*)(
                    const ZRegS &, const _PReg &, const ZRegS &);

            auto loop_mn = [this, load_loop_blk, ur](f &mn, PReg p, bool isSrc,
                                   ZRegS src = ZRegS(DUMMY_IDX)) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                    for (int i_ur = 0; i_ur < ur; ++i_ur) {
                        //auto r = vreg_accum(i_load, i_ur);
                        auto r = ZReg(i_ur * load_loop_blk + i_load);
                        if (isSrc)
                            (this->*mn)(r.s, p, src);
                        else
                            (this->*mn)(r.s, p, r.s);
                    }
            };

            f mn_fmaxnm = &CodeGenerator::fmaxnm;
            f mn_fminnm = &CodeGenerator::fminnm;
            f mn_frintn = &CodeGenerator::frintn;
            f mn_fcvtzs = &CodeGenerator::fcvtzs;

            if (jcp.dst_dt == data_type::u8)
                loop_mn(mn_fmaxnm, vmask, true, vmm_zero.s);
            loop_mn(mn_fminnm, vmask, true, vmm_saturation.s);
            loop_mn(mn_frintn, vmask, false);
            loop_mn(mn_fcvtzs, vmask, false);
#endif
        }

        // store to the destination
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                // const Vmm r_vmm = mask_flag ? r | ktail_mask : r;
                zmm_t r_vmm = r;

                auto base = aux_reg_output_data;
                auto raw_offt = jcp.typesize_out
                        * (jcp.oc_without_padding * i_ur
                                + i_load * jcp.load_block);

                assert(raw_offt <= INT_MAX);
                auto offt = static_cast<int>(raw_offt);

                int scale = 0;

                const int EVEX_max_8b_offt = 0x200;
                if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
                    offt = offt - 2 * EVEX_max_8b_offt;
                    scale = 1;
                } else if (3 * EVEX_max_8b_offt <= offt
                        && offt < 5 * EVEX_max_8b_offt) {
                    offt = offt - 4 * EVEX_max_8b_offt;
                    scale = 2;
                }

                auto re = offt;
                if (scale) re = re + (2 * EVEX_max_8b_offt) * scale;

                auto reg_tmp_adr = ((i_ur % 4) == 0)
                        ? reg_tmp0_adr
                        : ((i_ur % 4) == 1) ? reg_tmp1_adr
                                            : ((i_ur % 4) == 2) ? reg_tmp2_adr
                                                                : reg_tmp3_adr;
                auto reg_tmp_imm = ((i_ur % 4) == 0)
                        ? reg_tmp0_imm
                        : ((i_ur % 4) == 1) ? reg_tmp1_imm
                                            : ((i_ur % 4) == 2) ? reg_tmp2_imm
                                                                : reg_tmp3_imm;
                add_imm(reg_tmp_adr, XReg(base.getIdx()), re, reg_tmp_imm);

                auto _mask = mask_flag ? ktail_mask : vmask;
                switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        // vmovups(output_ptr(i_load, i_ur), r_vmm);
                        st1w(r_vmm.s, _mask, Xbyak_aarch64::ptr(reg_tmp_adr));
                        break;
                    case data_type::s8:
                        // vpmovsdb(output_ptr(i_load, i_ur), r_vmm);
                        smin(r_vmm.s, 127);
                        smax(r_vmm.s, -128);
                        st1b(r_vmm.s, _mask, Xbyak_aarch64::ptr(reg_tmp_adr));
                        break;
                    case data_type::u8:
                        // vpmovusdb(output_ptr(i_load, i_ur), r_vmm);
                        umin(r_vmm.s, 255);
                        st1b(r_vmm.s, _mask, Xbyak_aarch64::ptr(reg_tmp_adr));
                        break;
                    default: assert(!"unknown dst_dt");
                }
            }
        }
        ldr(reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        if (p_sum_scale && *p_sum_scale != 1.f)
            ldr(reg_load_data, SVE_compress_addr(reg_rsp, reg_load_data_off));
    };

    auto compute = [=](ZReg vreg_acc, ZReg vreg_wei, ZReg vreg_src) {
        // vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        sdot(ZRegS(vreg_acc.getIdx()), ZRegB(vreg_src.getIdx()),
                ZRegB(vreg_wei.getIdx()));
    };

    auto fma_block = [=](bool last_block) {
        int reduce_step = 4;
        int ic_tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
                ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
                : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                load_ptr(vreg_load(i_load), i_reduce, i_load);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (jcp.signed_input) {
                    if (last_block && ic_tail_size != 0
                            && i_reduce == loop_unroll - reduce_step) {
                        auto xmm_bcast = VReg16B(vmm_bcast.getIdx());
                        // load_bytes(xmm_bcast, aux_reg_bcast_data,
                        //         jcp.ic_without_padding * i_ur + i_reduce,
                        //         ic_tail_size);
                        for (int r = 0; r < ic_tail_size; ++r) {
                            add_imm(reg_tmp0_adr, aux_reg_bcast_data,
                                    (jcp.ic_without_padding * i_ur + i_reduce
                                            + r),
                                    reg_tmp0_imm);
                            ldrb(WReg(reg_tmp1_imm.getIdx()),
                                    Xbyak_aarch64::ptr(reg_tmp0_adr));
                            ins(VReg16B(xmm_bcast.getIdx())[r],
                                    WReg(reg_tmp1_imm.getIdx()));
                        }
                        // vpbroadcastd(vmm_bcast, xmm_bcast);
                        auto _bcast
                                = ((i_ur % 2) == 0) ? vmm_bcast : vmm_bcast2;
                        dup(ZRegS(_bcast.getIdx()),
                                ZRegS(xmm_bcast.getIdx())[0]);
                    } else {
                        if (i_ur == 0) {
                            // vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, i_ur, false));
                            bcast_ptr(vmm_bcast, i_reduce, i_ur, false);
                        }
                        if ((i_ur + 1) < ur) {
                            ZReg _bcast = ((i_ur % 2) == 0) ? vmm_bcast2
                                                            : vmm_bcast;
                            // vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, (i_ur+1), false));
                            bcast_ptr(_bcast, i_reduce, (i_ur + 1), false);
                        }
                    }
                    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                        ZReg _bcast
                                = ((i_ur % 2) == 0) ? vmm_bcast : vmm_bcast2;
                        compute(vreg_accum(i_load, i_ur), vreg_load(i_load),
                                _bcast);
                    }
                } else {
                    if (last_block && ic_tail_size != 0
                            && i_reduce == loop_unroll - reduce_step) {
                        auto xmm_bcast = VReg16B(vmm_bcast.getIdx());
                        for (int r = 0; r < ic_tail_size; ++r) {
                            add_imm(reg_tmp0_adr, aux_reg_bcast_data,
                                    (jcp.ic_without_padding * i_ur + i_reduce
                                            + r),
                                    reg_tmp0_imm);
                            ldrb(WReg(reg_tmp1_imm.getIdx()),
                                    Xbyak_aarch64::ptr(reg_tmp0_adr));
                            ins(VReg16B(xmm_bcast.getIdx())[r],
                                    WReg(reg_tmp1_imm.getIdx()));
                        }
                        dup(ZRegS(vmm_bcast.getIdx()),
                                ZRegS(xmm_bcast.getIdx())[0]);
                    } else {
                        bcast_ptr(vmm_bcast, i_reduce, i_ur, false);
                    }
                    xa_->add(vmm_bcast.b, vmm_bcast.b, vmm_shift.b);
                    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                        compute(vreg_accum(i_load, i_ur), vreg_load(i_load),
                                vmm_bcast);
                    }
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    xa_->mov(aux_reg_load_data, reg_load_data);

    xa_->mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    xa_->mov(reduce_loop_iter, reg_reduce_loop_work);
    subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
    b(LE, reduce_loop_tail);

    L(reduce_loop);
    {
        fma_block(false);
        adds(aux_reg_bcast_data, aux_reg_bcast_data,
                jcp.reduce_loop_bcast_step);
        adds(aux_reg_load_data, aux_reg_load_data, jcp.reduce_loop_load_step);
        subs(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll);
        b(GT, reduce_loop);
    }

    L(reduce_loop_tail);
    if (jcp.ic != jcp.ic_without_padding) {
        fma_block(true);
    } else {
        fma_block(false);
    }

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        str(reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));

        /*Check if it is the last load_loop_blk*/
        subs(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step);
        xa_->cmp_imm(reg_load_loop_work, 0, reg_tmp0_imm);
        b(GT, common_store);

        /*Check if it is the last ocb*/
        tst(reg_reduce_pos_flag, FLAG_OC_LAST);
        b(EQ, common_store);

        store(true);
        b(end_store);

        L(common_store);
        store(false);

        L(end_store);

        add_imm(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step, reg_tmp0_imm);
    } else {
        store(false);
    }
}

template <typename Vmm>
void _jit_sve_512_x8s8s32x_1x1_conv_kernel<Vmm>::generate() {

    preamble(true);
    const int simd_w = jcp.ic_block;

    ptrue(PRegB(vmask.getIdx()));
    pfalse(PRegB(mask_all_zero.getIdx()));

    // xor_(reg_scratch, reg_scratch);
    // Reg16 _t = reg_scratch.cvt16();
    // xa_->mov(_t, 0x1);
    // vpbroadcastw(vmm_one, _t);
    dup(vmm_one.h, 0x1);

    xa_->mov(reg_rsp, xa_->sp);
    subs(reg_rsp, reg_rsp, stack_space_needed);

    if (jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        // Reg32 regw_tmp = reg_last_load.cvt32();
        auto regw_tmp = reg_last_load;
        xa_->mov(regw_tmp, mask);
        // kmovw(ktail_mask, regw_tmp);
        index(ZRegS(0), 0, 1);
        xa_->mov(ZRegS(1), 1);
        lsl(ZRegS(1), vmask / Xbyak_aarch64::T_m, ZRegS(0));
        dup(ZRegS(0), WReg(regw_tmp.getIdx()));
        xa_->and_(ZRegD(0), ZRegD(0), ZRegD(1));
        cmpne(ktail_mask.s, vmask, ZRegS(0), 0);
    }

    if (jcp.with_bias)
        ldr(reg_bias_data,
                Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(bias_data)));
    if (!jcp.signed_input) {
        str(reg_bias_data, SVE_compress_addr(reg_rsp, reg_bias_data_off));
        ldr(reg_comp_data,
                Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(compensation)));
        str(reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
    }
#if 0
    if (jcp.src_zero_point) {
        ldr(reg_zp_compensation, Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(zp_compensation)));
        str(reg_zp_compensation, SVE_compress_addr(reg_rsp, reg_zp_compensation_off));
        ldr(reg_src_zero_point, Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(src_zero_point)));
        str(reg_src_zero_point, SVE_compress_addr(reg_rsp, reg_src_zero_point_off));
    }
    if (jcp.dst_zero_point) {
        ldr(reg_dst_zero_point, Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(dst_zero_point)));
        str(reg_dst_zero_point, SVE_compress_addr(reg_rsp, reg_dst_zero_point_off));
    }
#endif

    ldr(reg_ptr_scales, Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(scales)));
    str(reg_ptr_scales, SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
    ldr(reg_bcast_data,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(bcast_data)));
    ldr(reg_load_data, Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(load_data)));
    ldr(reg_output_data,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(output_data)));

    ldr(reg_load_loop_work,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(load_dim)));
    ldr(reg_bcast_loop_work,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(bcast_dim)));
    str(reg_bcast_loop_work, SVE_compress_addr(reg_rsp, bcast_loop_work_off));
    ldr(reg_reduce_loop_work,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(reduce_dim)));
    ldr(reg_reduce_pos_flag,
            Xbyak_aarch64::ptr(reg_abi_param1, GET_OFF(first_last_flag)));

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add_imm(reg_load_data, reg_load_data,
                load_loop_blk * jcp.load_loop_load_step, reg_tmp0_imm);
#if 0
	if (jcp.with_bias) {
            if (!jcp.signed_input)
                ldr(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
            add_imm(reg_bias_data, reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia,
                    reg_tmp0_imm);
            if (!jcp.signed_input)
                str(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
        }
        if (!jcp.signed_input) {
            ldr(reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
            add_imm(reg_comp_data, reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t),
                    reg_tmp0_imm);
            str(reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
        }
#else
        if (!jcp.signed_input) {
            if (jcp.with_bias) {
                ldr(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
                add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_bia,
                        reg_tmp0_imm);
                str(reg_bias_data,
                        SVE_compress_addr(reg_rsp, reg_bias_data_off));
            }
            ldr(reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
            add_imm(reg_comp_data, reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t),
                    reg_tmp0_imm);
            str(reg_comp_data, SVE_compress_addr(reg_rsp, reg_comp_data_off));
        } else {
            if (jcp.with_bias) {
                add_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_bia,
                        reg_tmp0_imm);
            }
        }
#endif
#if 0
        if (jcp.src_zero_point) {
            ldr(reg_zp_compensation,
                    SVE_compress_addr(reg_rsp, reg_zp_compensation_off));
            add_imm(reg_zp_compensation, reg_zp_compensation,
                    load_loop_blk * jcp.load_block * sizeof(int32_t), reg_tmp0_imm);
            str(reg_zp_compensation, SVE_compress_addr(reg_rsp, reg_zp_compensation_off));
        }
#endif
        str(reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        ldr(reg_ptr_scales, SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        add_imm(reg_ptr_scales, reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float),
                reg_tmp0_imm);
        str(reg_ptr_scales, SVE_compress_addr(reg_rsp, reg_ptr_sum_scale_off));
        ldr(reg_bcast_data, SVE_compress_addr(reg_rsp, reg_bcast_data_off));
        adds(reg_output_data, reg_output_data,
                load_loop_blk * jcp.load_block * jcp.typesize_out);
        subs(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step);
    };

    Label load_loop_blk[7];

    static const int ur_cases_fma_expl_bcast[] = {2, 5, 6, 9, 14, 32};
    const int size_ur_cases_fma = sizeof(ur_cases_fma_expl_bcast);
    const int *ur_cases_fma = ur_cases_fma_expl_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            xa_->cmp_imm(
                    reg_load_loop_work, simd_w * (label_idx + 1), reg_tmp0_imm);
            b(LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    xa_->cmp_imm(reg_load_loop_work, 0, reg_tmp0_imm);
                    b(EQ, load_loop_blk[num_ur_cases]);
                }

                for (int _i = 1; _i <= label_idx + 1; _i++) {
                    // prefetcht0(ptr[reg_load_data + _i * jcp.ic * jcp.oc_block]);
                    // prefetcht1(ptr[reg_output_data + _i * jcp.oc_block]);
                    add_imm(reg_tmp0_adr, reg_load_data,
                            (_i * jcp.ic * jcp.oc_block), reg_tmp0_imm);
                    add_imm(reg_tmp1_adr, reg_output_data, (_i * jcp.oc_block),
                            reg_tmp1_imm);
                    prfm(PLDL1KEEP, Xbyak_aarch64::ptr(reg_tmp0_adr));
                    prfm(PLDL2KEEP, Xbyak_aarch64::ptr(reg_tmp1_adr));
                }

                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    xa_->cmp_imm(reg_load_loop_work, 2 * label_idx * simd_w,
                            reg_tmp0_imm);
                    b(EQ, load_loop_blk[label_idx - 1]);
                }
                xa_->cmp_imm(reg_load_loop_work, (label_idx + 1) * simd_w,
                        reg_tmp0_imm);
                b(GE, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                xa_->cmp_imm(
                        reg_load_loop_work, simd_w * (idx + 1), reg_tmp0_imm);
                b(EQ, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                xa_->cmp_imm(reg_load_loop_work, simd_w, reg_tmp0_imm);
                b(LE, load_loop_blk[0]);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add_imm(reg_rsp, reg_rsp, stack_space_needed, reg_tmp0_imm);

    postamble();
}

bool jit_sve_512_x8s8s32x_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_convolution
            = [&](int idx) { return p.entry_[idx].is_convolution(); };

    int dw_idx = p.find(primitive_kind::convolution);
    int len = dw_idx != -1 ? dw_idx + 1 : p.len();

    switch (len) {
        case 0: return true;
        default: return false;
    }

    return false;
}

status_t jit_sve_512_x8s8s32x_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t *&src_md, memory_desc_t &weights_md,
        memory_desc_t &dst_md, memory_desc_t &bias_md,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    if (!mayiuse(sve_512)) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
        return status::unimplemented;

    jcp.nthr = nthreads;

    int ndims = src_d.ndims();

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.signed_input = (src_d.data_type() == data_type::s8);

    dim_t output_spatial = jcp.od * jcp.oh * jcp.ow;
    dim_t input_spatial = jcp.id * jcp.ih * jcp.iw;

    // FIXME: jcp.os and jcp.is fields have data type of int
    if (output_spatial > INT_MAX || input_spatial > INT_MAX)
        return status::unimplemented;

    jcp.os = output_spatial;
    jcp.is = input_spatial;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int dw_conv_ind = p.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.

#if 0
    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common
            = zp.common(DNNL_ARG_SRC); // otherwise, it's per-channel
    assert(IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common));

    if ((jcp.dst_zero_point || jcp.src_zero_point) && jcp.with_dw_conv)
        return status::unimplemented;
#endif

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, 16);
        jcp.ic = rnd_up(jcp.ic, 16);
    }

    const int simd_w = (jcp.ic % 16 == 0 && jcp.oc % 16 == 0)
            ? 16
            : (jcp.ic % 8 == 0 && jcp.oc % 8 == 0) ? 8 : 4;

    auto set_or_check_wei_format = [&]() -> bool {
        using namespace format_tag;
        using namespace memory_extra_flags;
        const format_tag_t wei_tags[3][2][3]
                = {{{OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i},
                           {gOIw4i16o4i, gOIhw4i16o4i, gOIdhw4i16o4i}},
                        {{OIw2i8o4i, OIhw2i8o4i, OIdhw2i8o4i},
                                {gOIw2i8o4i, gOIhw2i8o4i, gOIdhw2i8o4i}},
                        {{OIw4o4i, OIhw4o4i, OIdhw4o4i},
                                {gOIw4o4i, gOIhw4o4i, gOIdhw4o4i}}};

        const int simd_idx = simd_w == 16 ? 0 : simd_w == 8 ? 1 : 2;
        const auto wei_tag = wei_tags[simd_idx][with_groups][ndims - 3];
        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (!jcp.signed_input) {
            want_wei_md.extra.flags = 0 | compensation_conv_s8s8 | scale_adjust;
            want_wei_md.extra.compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust
                    // = mayiuse(avx512_core_vnni) ? 1.f : 0.5f;
                    = 1.f;
        }
#if 0
        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
        }
#endif

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_d == 1 && jcp.stride_h == 1
            && jcp.stride_w == 1 // TODO: support some strides
            && jcp.od == jcp.id && jcp.oh == jcp.ih
            && jcp.ow == jcp.iw // enforce rpad = 0
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;
    jcp.use_vmovntps = false;

    const int L2_size
            = platform::get_per_core_cache_size(2) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_treshold = 28;
    int max_regs = 0;
    int min_regs = 6;
    max_regs = ((jcp.oh > size_treshold && jcp.ow > size_treshold)
                       && (jcp.oc < 128 || jcp.ic < 128))
            ? min_regs
            : 9;
    jcp.expl_bcast = true;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_treshold && jcp.ow <= size_treshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs; // mobilenet_v2 performance improvement
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.od * jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
    }
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    jcp.reduce_dim = jcp.ic;
    jcp.reduce_block = jcp.ic_block;

    jcp.load_dim = jcp.oc;
    jcp.load_block = jcp.oc_block;

    jcp.bcast_dim = jcp.is;

    jcp.bcast_block = jcp.ur;

    jcp.reduce_loop_unroll = jcp.reduce_block;
    jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll * jcp.typesize_in;

    jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

    jcp.bcast_loop_output_step
            = jcp.ur * jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_output_substep = -1; // unused
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_bcast_substep = -1; // unused

    jcp.load_loop_load_step = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

    jcp.load_loop_iter_step = jcp.load_block;

    jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

    int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    reduce_blocking = nb_reduce;
    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 64;
    else if (jcp.bcast_dim > SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 16;
    reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
    reduce_blocking *= jcp.reduce_block;

    bool cmp_reduce = reduce_blocking <= jcp.reduce_dim;
    if (cmp_reduce) jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
    load_blocking = jcp.load_dim;

    jcp.load_grp_count = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            jcp.nthr, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= jcp.nthr
            && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2); //
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(jcp.nthr, jcp.load_grp_count))
            * jcp.bcast_block;
    bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
    bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

    int space_for_bcast = (L2_capacity - /* kernel_size - */
            2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
            - 3 * 1024);
    if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

    int bcast_in_cache
            = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
    bcast_blocking = nstl::min(
            bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

    load_blocking_max = load_blocking;
    bcast_blocking_max = bcast_blocking * 3 / 2;
    reduce_blocking_max = reduce_blocking;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.reduce_loop_unroll % 4 == 0);
    assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    // miniumum size of load dim chunk for work distribution within threads
    jcp.nb_load_chunk = 1;
    // peformance improvements for googlenet_v3, mb=1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    //    int ncores_per_socket = (int)cpu().getNumCores(
    //            Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    int ncores_per_socket = (int)2;
    if (jcp.mb == 1 && jcp.nb_load % 4 == 0 && jcp.ic / jcp.oc >= 4
            && jcp.ic * jcp.oc <= L2_size && jcp.nthr <= ncores_per_socket) {
        jcp.nb_load_chunk = 4;
        jcp.load_grp_count = nstl::max(jcp.nb_load / 4, jcp.load_grp_count);
    }

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // only common and per-oc-channel scales are supported
    const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
    if (!oscales_ok) return status::unimplemented;

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    if (jcp.dst_zero_point || jcp.src_zero_point) return status::unimplemented;

    if (jcp.ic_block != 16) return status::unimplemented;

    return status::success;
}

void jit_sve_512_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;
}

template struct _jit_sve_512_x8s8s32x_1x1_conv_kernel<ZReg>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
