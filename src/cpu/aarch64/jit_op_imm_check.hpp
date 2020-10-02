#ifndef JIT_OP_IMM_CHECK_HPP
#define JIT_OP_IMM_CHECK_HPP

#define LDRMAX    255
#define LDRMIN    (-256)
#define STRMAX    255
#define STRMIN    (-256)
#define LD1RWMAX  252
#define PRFMMAX   32760
#define PRFMMIN   0
#define PRFWMAX   31
#define PRFWMIN   (-32)

namespace dnnl{
namespace impl{
namespace cpu{
namespace aarch64{

// Load a vector register from a memory address generated by a 64-bit scalar base,
// plus an immediate offset in the range -256 to 255 which is multiplied 
// by the current vector register size in bytes. This instruction is unpredicated.
template <typename T>
bool ldr_imm_check(T ofs){
    int vlen = cpu_isa_traits<sve>::vlen;
    int vlen_shift = cpu_isa_traits<sve>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;
    return ((shifted_ofs) <= LDRMAX)
            && (shifted_ofs >= LDRMIN)
            && ((ofs % vlen) == 0);
}

// Store a vector register to a memory address generated by a 64-bit scalar base,
// plus an immediate offset in the range -256 to 255 which is multiplied
// by the current vector register size in bytes. This instruction is unpredicated.
template <typename T>
bool str_imm_check(T ofs){
    int vlen = cpu_isa_traits<sve>::vlen;
    int vlen_shift = cpu_isa_traits<sve>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;
    return ((shifted_ofs) <= STRMAX)
            && (shifted_ofs >= STRMIN)
            && ((ofs % vlen) == 0);
}
// Load a single unsigned word from a memory address generated by a 64-bit scalar
// base address plus an immediate offset which is a multiple of 4 in the range 0 to 252.
template <typename T>
bool ld1rw_imm_check(T ofs){
    return ((ofs & 0x3) == 0) && (ofs <= LD1RWMAX) && (ofs >= 0);
}

// Is the optional positive immediate byte offset, 
// a multiple of 8 in the range 0 to 32760, defaulting to 0 
// and encoded in the "imm12" field as <pimm>/8.
template <typename T>
bool prfm_imm_check(T ofs){
    return (ofs <= PRFMMAX) && (ofs >= PRFMMIN) && ((ofs & 0x7) == 0);
}

template <typename T>
bool prfw_imm_check(T ofs){
    int vlen = cpu_isa_traits<sve>::vlen;
    int vlen_shift = cpu_isa_traits<sve>::vlen_shift;
    int shifted_ofs = ofs >> vlen_shift;

    return (shifted_ofs <= PRFWMAX) && (shifted_ofs >= PRFWMIN) && ((ofs % vlen) == 0);
}


} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
