/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/jit_uni_eltwise_int.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak;
using namespace Xbyak_aarch64;

#define IDX(a) static_cast<uint32_t>(a.getIdx())

struct jit_args {
    const void *from;
    const void *for_comparison;
    const void *to;
    size_t work_amount;
};

struct jit_uni_eltwise_int_kernel : public c_compatible {
    jit_uni_eltwise_int_kernel(const eltwise_desc_t &desc) : desc_(desc) {}
    virtual ~jit_uni_eltwise_int_kernel() {}

    void operator()(const jit_args *args) {
        assert(ker_);
        ker_(args);
    }

protected:
    void (*ker_)(const jit_args *) = nullptr;

    data_type_t data_type() const { return desc_.data_desc.data_type; }
    int dtype_size() const { return types::data_type_size(data_type()); }

private:
    const eltwise_desc_t &desc_;
};

/* jit kernels */
namespace {
using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_uni_subkernel_int : public jit_uni_eltwise_int_kernel,
                               public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_subkernel_int)

    jit_uni_subkernel_int(const eltwise_desc_t &desc)
        : jit_uni_eltwise_int_kernel(desc), jit_generator() {
        using namespace data_type;

        // Relu and linear for int types: s32, s8, u8; Only forward direction
        assert(utils::one_of(desc.alg_kind, alg_kind::eltwise_relu,
                alg_kind::eltwise_linear));
        assert(utils::one_of(data_type(), s32, s8, u8));
#if 0
        assert(utils::one_of(isa, sse41, avx2, avx512_common));
#else
        assert(utils::one_of(isa, avx512_common));
#endif
        Reg64 param = abi_param1;

        const size_t vlen = cpu_isa_traits<isa>::vlen;
        const size_t simd_w = vlen / sizeof(float);
        const size_t loop_dec[] = {simd_w, 1};
        const size_t uf[] = {1, 1};
        const size_t shift[] = {dtype_size() * simd_w, (size_t)dtype_size()};
        const bool loop_vectorize[] = {true, false};

        preamble();

#define GET_OFF(field) offsetof(jit_args, field)
#if 0
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
#else
        CGA64::add_imm(X_TMP_0, XReg(IDX(param)), GET_OFF(from), X_TMP_1);
        CGA64::ldr(XReg(IDX(reg_from)), xa::ptr(X_TMP_0));

        CGA64::add_imm(X_TMP_0, XReg(IDX(param)), GET_OFF(to), X_TMP_1);
        CGA64::ldr(XReg(IDX(reg_to)), xa::ptr(X_TMP_0));

        CGA64::add_imm(
                X_TMP_0, XReg(IDX(param)), GET_OFF(work_amount), X_TMP_1);
        CGA64::ldr(XReg(IDX(reg_work_amount)), xa::ptr(X_TMP_0));
#endif
#undef GET_OFF

#if 0
        mov(imm_addr64, float2int(desc.alpha));
        uni_vmovq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);
#else
        std::cout << "Debug:jit_uni_subkernel_int L" << __LINE__ << std::endl;
        CGA64::mov_imm(WReg(IDX(X_TMP_0)), float2int(desc.alpha));
        CGA64::sxtw(XReg(IDX(imm_addr64)), WReg(IDX(X_TMP_0)));

        CGA64::bic(VReg(IDX(xmm_alpha)).b16, VReg(IDX(xmm_alpha)).b16,
                VReg(IDX(xmm_alpha)).b16);
        CGA64::mov(VReg(IDX(xmm_alpha)).d[0], XReg(IDX(imm_addr64)));

        //CGA64::dup(ZRegS(IDX(vmm_alpha)), ZRegS(IDX(xmm_alpha))[0]);
        CGA64::dup(vmm_alpha, ZRegS(IDX(xmm_alpha))[0]);
#endif

#if 0
        mov(imm_addr64, float2int(desc.beta));
        uni_vmovq(xmm_beta, imm_addr64);
        uni_vbroadcastss(vmm_beta, xmm_beta);
#else
        CGA64::mov_imm(WReg(IDX(X_TMP_0)), float2int(desc.beta));
        CGA64::sxtw(XReg(IDX(imm_addr64)), WReg(IDX(X_TMP_0)));

        CGA64::bic(VReg(IDX(xmm_beta)).b16, VReg(IDX(xmm_beta)).b16,
                VReg(IDX(xmm_beta)).b16);
        CGA64::mov(VReg(IDX(xmm_beta)).d[0], XReg(IDX(imm_addr64)));

        //CGA64::dup(ZRegS(IDX(vmm_beta)), ZRegS(IDX(xmm_beta))[0]);
        CGA64::dup(vmm_beta, ZRegS(IDX(xmm_beta))[0]);
#endif

#if 0
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        xor_(reg_int8, reg_int8);
#else
        CGA64::eor(ZRegD(IDX(vmm_zero)), ZRegD(IDX(vmm_zero)),
                ZRegD(IDX(vmm_zero)));
        CGA64::eor(
                XReg(IDX(reg_int8)), XReg(IDX(reg_int8)), XReg(IDX(reg_int8)));
#endif
        if (isa == avx512_common) {
#if 0
            mov(reg_int8.cvt8(), 0x01);
            kmovw(k_mask_int8, reg_int8.cvt32());
#else
            std::cout << "Debug:jit_uni_subkernel_int L" << __LINE__
                      << std::endl;
            CGA64::and_(X_TMP_1, XReg(IDX(reg_int8)), ~uint64_t(0xff));
            CGA64::mov(W_TMP_0, 0x01);
            CGA64::orr(XReg(IDX(reg_int8)), X_TMP_0, X_TMP_1);

            CGA64::mov(PRegB(IDX(k_mask_int8)), P_ALL_ONE.b);
            CGA64::index(ZRegS(IDX(z_tmp)), 0, 1);
            CGA64::mov(ZRegS(IDX(z_tmp0)), 1);
            CGA64::lsl(ZRegS(IDX(z_tmp0)), PReg(IDX(k_mask_int8)) / T_m,
                    ZRegS(IDX(z_tmp)));
            CGA64::dup(ZRegS(IDX(z_tmp)), WReg(IDX(reg_int8)));
            CGA64::and_(
                    ZRegD(IDX(z_tmp)), ZRegD(IDX(z_tmp)), ZRegD(IDX(z_tmp0)));
            CGA64::cmpne(PRegS(IDX(k_mask_int8)), PReg(IDX(k_mask_int8)),
                    ZRegS(IDX(z_tmp)), 0);
#endif
        }

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
#if 0
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
#else
            CGA64::mov_imm(X_TMP_0, uf[id] * loop_dec[id] - 1);
            CGA64::cmp(XReg(IDX(reg_work_amount)), X_TMP_0);
#endif
#if 0
            jle(loop_label[id + 1], T_NEAR);
#else
            CGA64::b(LE, loop_label[id + 1]);
#endif

            compute_step(loop_vectorize[id], uf[id], shift[id], desc.alg_kind);

#if 0
            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
#else
            CGA64::add_imm(XReg(IDX(reg_from)), XReg(IDX(reg_from)),
                    uf[id] * shift[id], X_TMP_0);
            CGA64::add_imm(XReg(IDX(reg_to)), XReg(IDX(reg_to)),
                    uf[id] * shift[id], X_TMP_0);
            CGA64::sub_imm(XReg(IDX(reg_work_amount)),
                    XReg(IDX(reg_work_amount)), uf[id] * loop_dec[id], X_TMP_0);
            CGA64::b(loop_label[id]);
#endif
        }

        L(loop_label[2]);
        postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    using opmask_t = const Xbyak::Opmask;

    XReg reg_from = XReg(0);
    XReg reg_to = XReg(8);
    Reg64 reg_work_amount = rsi; //x6
    Reg64 imm_addr64 = rbx; //x3
    Reg64 reg_int8 = r9;

    ZReg z_tmp {31};
    ZReg z_tmp0 {30};

    PReg p_tmp {0};

    XReg xmm_alpha = XReg(13);
    XReg xmm_beta = XReg(14);

    Vmm vmm_tmp = Vmm(isa == avx512_common ? 26 : 11);
    ZRegS vmm_alpha = ZRegS(27);
    ZRegS vmm_beta = ZRegS(28);
    Vmm vmm_zero = Vmm(29);
    //    Vmm vmm_mask = Vmm(isa == avx512_common ? 30 : 12);

    opmask_t k_mask = k1;
    opmask_t k_mask_int8 = k2; // Mask for store 1 byte in case of AVX512

    bool is32bit() const { return data_type() == data_type::s32; }

    // Load 32bit data type (s32)
    void load_32bit(
            const bool vectorize, const Vmm &vr_from, const XReg &mem_from) {

        if (vectorize) {
            // load full Vmm size
            std::cout << "Debug:load_32bit L" << __LINE__ << std::endl;
            CGA64::ldr(ZReg(IDX(vr_from)), xa::ptr(mem_from));
        } else {
            // load exactly one data item
            std::cout << "Debug:load_32bit L" << __LINE__ << std::endl;
            CGA64::ptrue(PRegS(IDX(p_tmp)), VL4);
            CGA64::ldr(W_TMP_0, xa::ptr(mem_from));
            CGA64::mov(ZRegS(IDX(Xmm(vr_from.getIdx()))),
                    PReg(IDX(p_tmp)) / T_m, 0);
            CGA64::ptrue(PRegS(IDX(p_tmp)), VL1);
            CGA64::mov(ZRegS(IDX(Xmm(vr_from.getIdx()))),
                    PReg(IDX(p_tmp)) / T_m, W_TMP_0);
        }
    }

    // Load 8bit data type (u8/s8)
    void load_8bit(const bool vectorize, const Vmm &vr_from,
            const XReg &mem_from, bool is_signed) {

        // data type u8/s8 load as s32
        if (vectorize) {
            // load full Vmm size
            if (is_signed) {
                std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
                CGA64::ldr(QReg(IDX(z_tmp)), xa::ptr(mem_from));
                CGA64::zip1(ZReg(IDX(z_tmp)).b, ZReg(IDX(z_tmp)).b,
                        ZReg(IDX(z_tmp)).b);
                CGA64::zip1(ZReg(IDX(z_tmp)).h, ZReg(IDX(z_tmp)).h,
                        ZReg(IDX(z_tmp)).h);
                CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
                CGA64::sxtb(ZReg(IDX(vr_from)).s, PReg(IDX(p_tmp)) / T_m,
                        ZReg(IDX(z_tmp)).s);
            } else {
                std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
                CGA64::ldr(QReg(IDX(z_tmp)), xa::ptr(mem_from));
                CGA64::zip1(ZReg(IDX(z_tmp)).b, ZReg(IDX(z_tmp)).b,
                        ZReg(IDX(z_tmp)).b);
                CGA64::zip1(ZReg(IDX(z_tmp)).h, ZReg(IDX(z_tmp)).h,
                        ZReg(IDX(z_tmp)).h);
                CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
                CGA64::uxtb(ZReg(IDX(vr_from)).s, PReg(IDX(p_tmp)) / T_m,
                        ZReg(IDX(z_tmp)).s);
            }
        } else {
            // load exactly one data item
            std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
            CGA64::ldurb(W_TMP_0, xa::ptr(mem_from));
            CGA64::and_(X_TMP_1, XReg(IDX(reg_int8.cvt8())), ~uint64_t(0xff));
            CGA64::orr(XReg(IDX(reg_int8.cvt8())), X_TMP_0, X_TMP_1);

            if (is_signed) {
                std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
                CGA64::sxtb(WReg(IDX(reg_int8.cvt32())),
                        WReg(IDX(reg_int8.cvt8())));
            } else {
                std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
                CGA64::uxtb(WReg(IDX(reg_int8.cvt32())),
                        WReg(IDX(reg_int8.cvt8())));
            }

            std::cout << "Debug:load_8bit L" << __LINE__ << std::endl;
            CGA64::bic(VReg(IDX(Xmm(vr_from.getIdx()))).b16,
                    VReg(IDX(Xmm(vr_from.getIdx()))).b16,
                    VReg(IDX(Xmm(vr_from.getIdx()))).b16);
            CGA64::mov(
                    VReg(IDX(Xmm(vr_from.getIdx()))).d[0], XReg(IDX(reg_int8)));
        }
    }

    // Load vregs with data from mem
    void load(const bool vectorize, const Vmm &vr_from, const XReg &mem_from) {

        // Branching on data size
        if (is32bit())
            load_32bit(vectorize, vr_from, mem_from);
        else
            load_8bit(
                    vectorize, vr_from, mem_from, data_type() == data_type::s8);
    }

    // Processing
    void process_linear(const Vmm &vr_to, const Vmm &vr_from);
    void process_relu(const Vmm &vr_to, const Vmm &vr_from);

    // Store s32 for any isa
    void store_32bit(
            const bool vectorize, const XReg &mem_to, const Vmm &vr_to) {
        if (vectorize) {
            // store full Vmm size
            std::cout << "Debug:store_32bit L" << __LINE__ << std::endl;
            CGA64::str(ZReg(IDX(vr_to)), xa::ptr(mem_to));
        } else {
            // store exactly one data item
            std::cout << "Debug:store_32bit L" << __LINE__ << std::endl;
            CGA64::ptrue(PRegS(IDX(p_tmp)), VL1);
            CGA64::st1w(ZRegS(IDX(Xmm(vr_to.getIdx()))), PReg(IDX(p_tmp)),
                    xa::ptr(mem_to));
        }
    }

    // Store 8 bit int - isa-dependent
    void store_8bit(const bool vectorize, const XReg &mem_to, const Vmm &vr_to,
            bool is_signed);

    // Store results from vregs to mem
    void store(const bool vectorize, const XReg &mem_to, const Vmm &vr_to) {
        // Branching on data size
        if (is32bit())
            store_32bit(vectorize, mem_to, vr_to);
        else
            store_8bit(vectorize, mem_to, vr_to, data_type() == data_type::s8);
    }

    void compute_step(bool vectorize, const size_t uf, const size_t shift,
            const alg_kind_t alg) {

        auto vreg_from = [&](const size_t i) -> Vmm { return Vmm(i + 1); };
        auto vreg_to = [&](const size_t i) -> Vmm { return Vmm(uf + i + 1); };

        std::cout << "Debug:compute_step / uf = " << uf << std::endl;
        // 1. Load (vregs <- mem)
        for (size_t i = 0; i < uf; i++) {
            CGA64::add_imm(reg_from, reg_from, i * shift, X_TMP_0);
            load(vectorize, vreg_from(i), reg_from);
        }

        // 2. Process (vregs <- vergs)
        switch (alg) {
            case alg_kind::eltwise_linear:
                for (size_t i = 0; i < uf; i++)
                    process_linear(vreg_to(i), vreg_from(i));
                break;
            case alg_kind::eltwise_relu:
                for (size_t i = 0; i < uf; i++)
                    process_relu(vreg_to(i), vreg_from(i));
                break;
            default: assert(!"unsupported alg");
        }

        // 3. Store (mem <- vregs)
        for (size_t i = 0; i < uf; i++) {
            CGA64::add_imm(reg_to, reg_to, i * shift, X_TMP_0);
            store(vectorize, reg_to, vreg_to(i));
        }
    }
};

template <cpu_isa_t isa>
void jit_uni_subkernel_int<isa>::process_linear(
        const Vmm &vr_to, const Vmm &vr_from) {
    std::cout << "Debug:process_linear L" << __LINE__ << std::endl;

    CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
    CGA64::scvtf(
            ZReg(IDX(vr_to)).s, PReg(IDX(p_tmp)) / T_m, ZReg(IDX(vr_from)).s);

    // ###Manually merge code generated by the indirect method
    CGA64::sub_imm(XReg(22), XReg(22), 0x08, X_TMP_0);
    CGA64::str(PReg(7), xa::ptr(XReg(22)));
    CGA64::mov(PReg(7).b, PReg(15).b);
    //CGA64::fmad(ZReg(2).s,PReg(7)/T_m,ZReg(27).s,ZReg(28).s);
    CGA64::fmad(ZReg(2).s, PReg(7) / T_m, vmm_alpha, vmm_beta);
    CGA64::ldr(PReg(7), xa::ptr(XReg(22)));
    CGA64::add_imm(XReg(22), XReg(22), 0x08, X_TMP_0);

    // Saturate before converting from f32 to s32
    Vmm vmm_saturation_ubound = vmm_tmp;
    Reg64 reg_tmp = r10;
    CGA64::eor(
            ZRegD(IDX(vmm_zero)), ZRegD(IDX(vmm_zero)), ZRegD(IDX(vmm_zero)));
    init_saturate_f32(vmm_zero, vmm_saturation_ubound, reg_tmp, data_type::f32,
            data_type());
    saturate_f32(vr_to, vmm_zero, vmm_saturation_ubound, data_type());

    CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE / xa::T_z, P_ALL_ONE.b);
    CGA64::frinti(ZRegS(IDX(vr_to)), PReg(IDX(p_tmp)) / T_m, ZRegS(IDX(vr_to)));
    CGA64::fcvtzs(ZRegS(IDX(vr_to)), PReg(IDX(p_tmp)) / T_m, ZRegS(IDX(vr_to)));
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int<isa>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {
    assert(!"unsupported isa");
}

#if 0
template <>
void jit_uni_subkernel_int<sse41>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    cvtdq2ps(vr_from, vr_from);
    movups(vr_to, vr_from);
    mulps(vr_to, vmm_alpha);

    Vmm mask = Vmm(0);
    movups(mask, vr_from);
    cmpps(mask, vmm_zero, _cmp_nle_us);
    blendvps(vr_to, vr_from);
    cvtps2dq(vr_to, vr_to);
}

template <>
void jit_uni_subkernel_int<avx2>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    vcvtdq2ps(vr_from, vr_from);
    vmulps(vr_to, vr_from, vmm_alpha);
    vcmpgtps(vmm_mask, vr_from, vmm_zero);
    vblendvps(vr_to, vr_to, vr_from, vmm_mask);
    vcvtps2dq(vr_to, vr_to);
}
#endif

template <>
void jit_uni_subkernel_int<avx512_common>::process_relu(
        const Vmm &vr_to, const Vmm &vr_from) {

    std::cout << "Debug:process_relu L" << __LINE__ << std::endl;
    CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
    CGA64::scvtf(
            ZReg(IDX(vr_from)).s, PReg(IDX(p_tmp)) / T_m, ZReg(IDX(vr_from)).s);

    CGA64::fmul(ZReg(IDX(vr_to)).s, ZReg(IDX(vr_from)).s, vmm_alpha);

    CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE / xa::T_z, P_ALL_ONE.b);
    CGA64::fcmgt(PRegS(IDX(k_mask)), PReg(IDX(p_tmp)) / xa::T_z,
            ZRegS(IDX(vr_from)), ZRegS(IDX(vmm_zero)));

    CGA64::sel(ZRegS(IDX(vr_to)), PReg(IDX(k_mask)) / T_m, ZRegS(IDX(vr_from)),
            ZRegS(IDX(vr_to)));

    CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE / xa::T_z, P_ALL_ONE.b);
    CGA64::frinti(ZRegS(IDX(vr_to)), PReg(IDX(p_tmp)) / T_m, ZRegS(IDX(vr_to)));
    CGA64::fcvtzs(ZRegS(IDX(vr_to)), PReg(IDX(p_tmp)) / T_m, ZRegS(IDX(vr_to)));
}

template <cpu_isa_t isa>
void jit_uni_subkernel_int<isa>::store_8bit(const bool vectorize,
        const XReg &mem_to, const Vmm &vr_to, bool is_signed) {
    assert(!"unsupported isa");
}

#if 0
template <>
void jit_uni_subkernel_int<sse41>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16
        packssdw(vr_to, vmm_zero);
        // s16 -> s8/u8
        if (is_signed)
            packsswb(vr_to, vmm_zero);
        else
            packuswb(vr_to, vmm_zero);

        movd(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        packssdw(vr_to, vmm_zero);
        if (is_signed)
            packsswb(vr_to, vmm_zero);
        else
            packuswb(vr_to, vmm_zero);
        movd(reg_int8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_int8.cvt8());
    }
}

template <>
void jit_uni_subkernel_int<avx2>::store_8bit(const bool vectorize,
        const Address &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        // s32 -> s16 = {qw0, 0, qw1, 0}
        vpackssdw(vr_to, vr_to, vmm_zero);
        // permute to restore order{qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
        vpermq(Ymm(vr_to.getIdx()), Ymm(vr_to.getIdx()), 0x58);

        // s16 -> s8/u8 : {16 x s16}{16 x 0} -> {32 x s8/u8}
        if (is_signed)
            vpacksswb(vr_to, vr_to, vmm_zero);
        else
            vpackuswb(vr_to, vr_to, vmm_zero);
        uni_vmovq(mem_to, Xmm(vr_to.getIdx()));
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        vpackssdw(vr_to, vr_to, vmm_zero);
        if (is_signed)
            vpacksswb(vr_to, vr_to, vmm_zero);
        else
            vpackuswb(vr_to, vr_to, vmm_zero);
        vmovd(reg_int8.cvt32(), Xmm(vr_to.getIdx()));
        mov(mem_to, reg_int8.cvt8());
    }
}
#endif

template <>
void jit_uni_subkernel_int<avx512_common>::store_8bit(const bool vectorize,
        const XReg &mem_to, const Vmm &vr_to, bool is_signed) {
    if (vectorize) {
        // store full Vmm size
        if (is_signed) {
            std::cout << "Debug:store_8bit L" << __LINE__ << std::endl;
            CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
            CGA64::mov(ZRegD(IDX(z_tmp)), ZRegD(IDX(vr_to)));
            CGA64::smin(ZRegS(IDX(z_tmp)), 127);
            CGA64::smax(ZRegS(IDX(z_tmp)), -128);
            CGA64::st1b(ZRegS(IDX(z_tmp)), PReg(IDX(p_tmp)), xa::ptr(mem_to));
        } else {
            std::cout << "Debug:store_8bit L" << __LINE__ << std::endl;
            CGA64::mov(PRegB(IDX(p_tmp)), P_ALL_ONE.b);
            CGA64::mov(ZRegD(IDX(z_tmp)), ZRegD(IDX(vr_to)));
            CGA64::umin(ZRegS(IDX(z_tmp)), 255);
            CGA64::st1b(ZRegS(IDX(z_tmp)), PReg(IDX(p_tmp)), xa::ptr(mem_to));
        }
    } else {
        // store exactly one data item
        // s32 save as s8/u8
        if (is_signed) {
            std::cout << "Debug:store_8bit L" << __LINE__ << std::endl;
            CGA64::mov(ZRegD(IDX(z_tmp)), ZRegD(IDX(vr_to)));
            CGA64::smin(ZRegS(IDX(z_tmp)), 127);
            CGA64::smax(ZRegS(IDX(z_tmp)), -128);
            CGA64::st1b(
                    ZRegS(IDX(z_tmp)), PReg(IDX(k_mask_int8)), xa::ptr(mem_to));
        } else {
            std::cout << "Debug:store_8bit L" << __LINE__ << std::endl;
            CGA64::mov(ZRegD(IDX(z_tmp)), ZRegD(IDX(vr_to)));
            CGA64::umin(ZRegS(IDX(z_tmp)), 255);
            CGA64::st1b(
                    ZRegS(IDX(z_tmp)), PReg(IDX(k_mask_int8)), xa::ptr(mem_to));
        }
    }
}

} /* namespace */

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_int_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    bool ok = mayiuse(isa)
            && desc()->data_desc.data_type == d_type
            // only relu and linear so far
            && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu,
                    alg_kind::eltwise_linear)
            && !has_zero_dim_memory()
            && memory_desc_wrapper(src_md()).is_dense(true)
            && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::jit_uni_eltwise_int_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {
    const auto &desc = *pd()->desc();
    kernel_ = new jit_uni_subkernel_int<isa>(desc);
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_int_fwd_t<isa, d_type>::~jit_uni_eltwise_int_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
void jit_uni_eltwise_int_fwd_t<isa, d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const size_t nelems = data_d.nelems(true);

    src += data_d.offset0();
    dst += data_d.offset0();

    const int cache_line = 64 / data_d.data_type_size();
    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = (const void *)&src[start];
        arg.for_comparison = (const void *)&src[start];
        arg.to = (const void *)&dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount) (*kernel_)(&arg);
    });
}

using namespace data_type;

//template struct jit_uni_eltwise_int_fwd_t<sse41, s32>;
//template struct jit_uni_eltwise_int_fwd_t<avx2, s32>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s32>;

//template struct jit_uni_eltwise_int_fwd_t<sse41, s8>;
//template struct jit_uni_eltwise_int_fwd_t<avx2, s8>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, s8>;

//template struct jit_uni_eltwise_int_fwd_t<sse41, u8>;
//template struct jit_uni_eltwise_int_fwd_t<avx2, u8>;
template struct jit_uni_eltwise_int_fwd_t<avx512_common, u8>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
