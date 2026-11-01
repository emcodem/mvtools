// Shared declarations of the mvtools_vulkan compute shaders. Every shader starts with
//   #version 460
//   #extension GL_GOOGLE_include_directive : require
//   #include "common.glsl"
// Device memory is reached through 64-bit addresses (uint64_t) with GL_EXT_buffer_reference: PU8(addr).v[i] etc.
// The kernel's argument struct (same layout as the C struct: scalar layout, only 32/64-bit members) comes through the push
// constant: declare `struct Args { ... };`, then `ARGS(Args)`, then read the members as `A.member`.
// Files named x.pix.comp are compiled twice: with MVC_PIX8 (8-bit samples, x_8) and without (16-bit samples, x_16).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : enable

layout(buffer_reference, scalar, buffer_reference_align = 1) buffer PU8 { uint8_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer PI8 { int8_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer PU16 { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer PI16 { int16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PU32 { uint v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PI32 { int v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PF32 { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer PU64 { uint64_t v[]; };

// device-coherent views for data other workgroups of the same dispatch write (spin waits, tickets); use with
// memoryBarrierBuffer() / atomics
layout(buffer_reference, scalar, buffer_reference_align = 4) coherent buffer PI32C { int v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) coherent volatile buffer PI32V { int v[]; };

#define ARGS(T) \
  layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer ArgsRef_ { T a; }; \
  layout(push_constant, scalar) uniform PushC_ { ArgsRef_ ref; } pc_;
#define A (pc_.ref.a)

// sample access of the compile-time sample size (x.pix.comp)
#ifdef MVC_PIX8
#define PIX_BYTES 1
#define PPIX PU8
#define pix_t uint8_t
#else
#define PIX_BYTES 2
#define PPIX PU16
#define pix_t uint16_t
#endif

// sample i (in samples) of the plane at address base
int ldpix(uint64_t base, uint i) { return int(uint(PPIX(base).v[i])); }
void stpix(uint64_t base, uint i, int v) { PPIX(base).v[i] = pix_t(uint(v)); }

// explicit-size access when one shader handles both sizes at run time (bytes = 1 or 2)
int ld8(uint64_t base, uint i) { return int(uint(PU8(base).v[i])); }
int ld16(uint64_t base, uint i) { return int(uint(PU16(base).v[i])); }
void st8(uint64_t base, uint i, int v) { PU8(base).v[i] = uint8_t(uint(v)); }
void st16(uint64_t base, uint i, int v) { PU16(base).v[i] = uint16_t(uint(v)); }
int ldb(uint64_t base, uint i, int bytes) { return bytes == 1 ? ld8(base, i) : ld16(base, i); }
void stb(uint64_t base, uint i, int v, int bytes) { if (bytes == 1) st8(base, i, v); else st16(base, i, v); }
