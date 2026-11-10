// fftp_pow2: the complex FFT of pocketfft (cfftp: radix-8 / 4 / 2 passes, its twiddle tables) for power-of-two
// lengths, restructured so one pass can run as independent work items (CPU loops or GPU threads). It reproduces
// pocketfft bit for bit (same operations in the same order, same twiddles), which is the FFT numpy uses; SplatFps'
// phase correlation needs that to make the same peak decisions as its numpy reference where surface values tie.
// Requirements: float arithmetic in source order (no contraction into FMA), IEEE double for the twiddle tables.
//
// Derived from pocketfft (https://github.com/mreineck/pocketfft, C++ branch, pocketfft_hdronly.h), whose licence
// follows:
//
// Copyright (C) 2010-2022 Max-Planck-Society
// Copyright (C) 2019-2020 Peter Bell
// Authors: Martin Reinecke, Peter Bell
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice, this
//   list of conditions and the following disclaimer in the documentation and/or
//   other materials provided with the distribution.
// * Neither the name of the copyright holder nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#ifdef __CUDACC__
#define FFTP_HD __host__ __device__ __forceinline__
#else
#define FFTP_HD inline
#endif

namespace fftp
{
  template <typename T> struct cpx { T r, i; };

  template <typename T> FFTP_HD cpx<T> operator+(cpx<T> a, cpx<T> b) { return { a.r + b.r, a.i + b.i }; }
  template <typename T> FFTP_HD cpx<T> operator-(cpx<T> a, cpx<T> b) { return { a.r - b.r, a.i - b.i }; }
  template <typename T> FFTP_HD void PM(cpx<T> &a, cpx<T> &b, cpx<T> c, cpx<T> d) { a = c + d; b = c - d; }
  template <typename T> FFTP_HD void PMINPLACE(cpx<T> &a, cpx<T> &b) { cpx<T> t = a; a.r += b.r; a.i += b.i; b = t - b; }
  template <bool fwd, typename T> FFTP_HD void special_mul(cpx<T> v1, cpx<T> v2, cpx<T> &res)
  {
    res = fwd ? cpx<T>{ v1.r * v2.r + v1.i * v2.i, v1.i * v2.r - v1.r * v2.i }
              : cpx<T>{ v1.r * v2.r - v1.i * v2.i, v1.r * v2.i + v1.i * v2.r };
  }
  template <bool fwd, typename T> FFTP_HD void ROTX90(cpx<T> &a)
  {
    T tmp = fwd ? -a.r : a.r;
    a.r = fwd ? a.i : -a.i;
    a.i = tmp;
  }
  template <bool fwd, typename T> FFTP_HD void ROTX45(cpx<T> &a)
  {
    const T h = (T)0.707106781186547524400844362104849;
    T tmp = a.r;
    if (fwd)
    {
      a.r = h * (a.r + a.i);
      a.i = h * (a.i - tmp);
    }
    else
    {
      a.r = h * (a.r - a.i);
      a.i = h * (a.i + tmp);
    }
  }
  template <bool fwd, typename T> FFTP_HD void ROTX135(cpx<T> &a)
  {
    const T h = (T)0.707106781186547524400844362104849;
    T tmp = a.r;
    if (fwd)
    {
      a.r = h * (a.i - a.r);
      a.i = h * (-tmp - a.i);
    }
    else
    {
      a.r = h * (-a.r - a.i);
      a.i = h * (tmp - a.i);
    }
  }

  // ---- one work item (k, i) of a pass: cc = input, ch = output (length n), wa = this pass's twiddles
  template <bool fwd, typename T>
  FFTP_HD void pass2_item(int ido, int l1, int k, int i, const cpx<T> *cc, cpx<T> *ch, const cpx<T> *wa)
  {
    auto CH = [&](int a, int b, int c) -> cpx<T> & { return ch[a + ido * (b + l1 * c)]; };
    auto CC = [&](int a, int b, int c) -> cpx<T> { return cc[a + ido * (b + 2 * c)]; };
    if (i == 0)
    {
      CH(0, k, 0) = CC(0, 0, k) + CC(0, 1, k);
      CH(0, k, 1) = CC(0, 0, k) - CC(0, 1, k);
    }
    else
    {
      CH(i, k, 0) = CC(i, 0, k) + CC(i, 1, k);
      special_mul<fwd>(CC(i, 0, k) - CC(i, 1, k), wa[i - 1], CH(i, k, 1));
    }
  }

  template <bool fwd, typename T>
  FFTP_HD void pass4_item(int ido, int l1, int k, int i, const cpx<T> *cc, cpx<T> *ch, const cpx<T> *wa)
  {
    auto CH = [&](int a, int b, int c) -> cpx<T> & { return ch[a + ido * (b + l1 * c)]; };
    auto CC = [&](int a, int b, int c) -> cpx<T> { return cc[a + ido * (b + 4 * c)]; };
    auto WA = [&](int x, int ii) { return wa[ii - 1 + x * (ido - 1)]; };
    cpx<T> t1, t2, t3, t4;
    if (i == 0)
    {
      PM(t2, t1, CC(0, 0, k), CC(0, 2, k));
      PM(t3, t4, CC(0, 1, k), CC(0, 3, k));
      ROTX90<fwd>(t4);
      PM(CH(0, k, 0), CH(0, k, 2), t2, t3);
      PM(CH(0, k, 1), CH(0, k, 3), t1, t4);
    }
    else
    {
      cpx<T> cc0 = CC(i, 0, k), cc1 = CC(i, 1, k), cc2 = CC(i, 2, k), cc3 = CC(i, 3, k);
      PM(t2, t1, cc0, cc2);
      PM(t3, t4, cc1, cc3);
      ROTX90<fwd>(t4);
      CH(i, k, 0) = t2 + t3;
      special_mul<fwd>(t1 + t4, WA(0, i), CH(i, k, 1));
      special_mul<fwd>(t2 - t3, WA(1, i), CH(i, k, 2));
      special_mul<fwd>(t1 - t4, WA(2, i), CH(i, k, 3));
    }
  }

  template <bool fwd, typename T>
  FFTP_HD void pass8_item(int ido, int l1, int k, int i, const cpx<T> *cc, cpx<T> *ch, const cpx<T> *wa)
  {
    auto CH = [&](int a, int b, int c) -> cpx<T> & { return ch[a + ido * (b + l1 * c)]; };
    auto CC = [&](int a, int b, int c) -> cpx<T> { return cc[a + ido * (b + 8 * c)]; };
    auto WA = [&](int x, int ii) { return wa[ii - 1 + x * (ido - 1)]; };
    cpx<T> a0, a1, a2, a3, a4, a5, a6, a7;
    if (i == 0)
    {
      PM(a1, a5, CC(0, 1, k), CC(0, 5, k));
      PM(a3, a7, CC(0, 3, k), CC(0, 7, k));
      PMINPLACE(a1, a3);
      ROTX90<fwd>(a3);

      ROTX90<fwd>(a7);
      PMINPLACE(a5, a7);
      ROTX45<fwd>(a5);
      ROTX135<fwd>(a7);

      PM(a0, a4, CC(0, 0, k), CC(0, 4, k));
      PM(a2, a6, CC(0, 2, k), CC(0, 6, k));
      PM(CH(0, k, 0), CH(0, k, 4), a0 + a2, a1);
      PM(CH(0, k, 2), CH(0, k, 6), a0 - a2, a3);
      ROTX90<fwd>(a6);
      PM(CH(0, k, 1), CH(0, k, 5), a4 + a6, a5);
      PM(CH(0, k, 3), CH(0, k, 7), a4 - a6, a7);
    }
    else
    {
      PM(a1, a5, CC(i, 1, k), CC(i, 5, k));
      PM(a3, a7, CC(i, 3, k), CC(i, 7, k));
      ROTX90<fwd>(a7);
      PMINPLACE(a1, a3);
      ROTX90<fwd>(a3);
      PMINPLACE(a5, a7);
      ROTX45<fwd>(a5);
      ROTX135<fwd>(a7);
      PM(a0, a4, CC(i, 0, k), CC(i, 4, k));
      PM(a2, a6, CC(i, 2, k), CC(i, 6, k));
      PMINPLACE(a0, a2);
      CH(i, k, 0) = a0 + a1;
      special_mul<fwd>(a0 - a1, WA(3, i), CH(i, k, 4));
      special_mul<fwd>(a2 + a3, WA(1, i), CH(i, k, 2));
      special_mul<fwd>(a2 - a3, WA(5, i), CH(i, k, 6));
      ROTX90<fwd>(a6);
      PMINPLACE(a4, a6);
      special_mul<fwd>(a4 + a5, WA(0, i), CH(i, k, 1));
      special_mul<fwd>(a4 - a5, WA(4, i), CH(i, k, 5));
      special_mul<fwd>(a6 + a7, WA(2, i), CH(i, k, 3));
      special_mul<fwd>(a6 - a7, WA(6, i), CH(i, k, 7));
    }
  }

  // ---- plan: factors (8s first, then 4s, a single 2 moved to the front), per pass ido / l1 and twiddle offset
  constexpr int MAXPASS = 8;
  struct Plan
  {
    int n = 0, npass = 0;
    int ip[MAXPASS], ido[MAXPASS], l1[MAXPASS], twofs[MAXPASS];
  };

  template <bool fwd, typename T>
  FFTP_HD void pass_item(const Plan &p, int s, int item, const cpx<T> *cc, cpx<T> *ch, const cpx<T> *tw)
  {
    // item = k * ido + i
    const int ido = p.ido[s], l1 = p.l1[s], k = item / ido, i = item % ido;
    const cpx<T> *wa = tw + p.twofs[s];
    if (p.ip[s] == 8)
      pass8_item<fwd>(ido, l1, k, i, cc, ch, wa);
    else if (p.ip[s] == 4)
      pass4_item<fwd>(ido, l1, k, i, cc, ch, wa);
    else
      pass2_item<fwd>(ido, l1, k, i, cc, ch, wa);
  }

  // pocketfft's twiddle source: exp(+-2 pi i k / n) from two tables of octant-reduced cos / sin in double, multiplied
  class SinCos
  {
    size_t N, mask, shift;
    std::vector<cpx<double>> v1, v2;

    static cpx<double> calc(size_t x, size_t n, double ang)
    {
      x <<= 3;
      if (x < 4 * n)
      {
        if (x < 2 * n)
        {
          if (x < n)
            return { std::cos(double(x) * ang), std::sin(double(x) * ang) };
          return { std::sin(double(2 * n - x) * ang), std::cos(double(2 * n - x) * ang) };
        }
        x -= 2 * n;
        if (x < n)
          return { -std::sin(double(x) * ang), std::cos(double(x) * ang) };
        return { -std::cos(double(2 * n - x) * ang), std::sin(double(2 * n - x) * ang) };
      }
      x = 8 * n - x;
      if (x < 2 * n)
      {
        if (x < n)
          return { std::cos(double(x) * ang), -std::sin(double(x) * ang) };
        return { std::sin(double(2 * n - x) * ang), -std::cos(double(2 * n - x) * ang) };
      }
      x -= 2 * n;
      if (x < n)
        return { -std::sin(double(x) * ang), -std::cos(double(x) * ang) };
      return { -std::cos(double(2 * n - x) * ang), -std::sin(double(2 * n - x) * ang) };
    }

  public:
    explicit SinCos(size_t n) : N(n)
    {
      const double pi = 3.141592653589793238462643383279502884197;
      const double ang = 0.25 * pi / (double)n;
      const size_t nval = (n + 2) / 2;
      shift = 1;
      while ((size_t(1) << shift) * (size_t(1) << shift) < nval)
        ++shift;
      mask = (size_t(1) << shift) - 1;
      v1.resize(mask + 1);
      v1[0] = { 1.0, 0.0 };
      for (size_t i = 1; i < v1.size(); ++i)
        v1[i] = calc(i, n, ang);
      v2.resize((nval + mask) / (mask + 1));
      v2[0] = { 1.0, 0.0 };
      for (size_t i = 1; i < v2.size(); ++i)
        v2[i] = calc(i * (mask + 1), n, ang);
    }

    template <typename T> cpx<T> get(size_t idx) const
    {
      if (2 * idx <= N)
      {
        const cpx<double> x1 = v1[idx & mask], x2 = v2[idx >> shift];
        return { T(x1.r * x2.r - x1.i * x2.i), T(x1.r * x2.i + x1.i * x2.r) };
      }
      idx = N - idx;
      const cpx<double> x1 = v1[idx & mask], x2 = v2[idx >> shift];
      return { T(x1.r * x2.r - x1.i * x2.i), -T(x1.r * x2.i + x1.i * x2.r) };
    }
  };

  // plan and twiddles of length n (a power of two >= 2) for element type T
  template <typename T> Plan make_plan(int n, std::vector<cpx<T>> &tw)
  {
    Plan p;
    p.n = n;
    int len = n, fct[MAXPASS], nf = 0;
    while ((len & 7) == 0)
    {
      fct[nf++] = 8;
      len >>= 3;
    }
    while ((len & 3) == 0)
    {
      fct[nf++] = 4;
      len >>= 2;
    }
    if ((len & 1) == 0)
    {
      len >>= 1;
      fct[nf++] = 2;
      const int t = fct[0];
      fct[0] = fct[nf - 1];
      fct[nf - 1] = t;
    }
    p.npass = nf;
    const SinCos sc((size_t)n);
    tw.clear();
    int l1 = 1;
    for (int k = 0; k < nf; k++)
    {
      const int ip = fct[k], ido = n / (l1 * ip);
      p.ip[k] = ip;
      p.ido[k] = ido;
      p.l1[k] = l1;
      p.twofs[k] = (int)tw.size();
      tw.resize(tw.size() + (size_t)(ip - 1) * (ido - 1));
      for (int j = 1; j < ip; ++j)
        for (int i = 1; i < ido; ++i)
          tw[p.twofs[k] + (j - 1) * (ido - 1) + i - 1] = sc.get<T>((size_t)j * l1 * i);
      l1 *= ip;
    }
    if (tw.empty())
      tw.push_back({ T(1), T(0) });
    return p;
  }

  // the whole transform of one line on the CPU: c in / out, ch scratch (n each); fct scales the result when != 1
  template <bool fwd, typename T>
  void exec(const Plan &p, cpx<T> *c, cpx<T> *ch, const cpx<T> *tw, T fct)
  {
    cpx<T> *p1 = c, *p2 = ch;
    for (int s = 0; s < p.npass; s++)
    {
      const int items = p.l1[s] * p.ido[s];
      for (int it = 0; it < items; it++)
        pass_item<fwd>(p, s, it, p1, p2, tw);
      cpx<T> *t = p1;
      p1 = p2;
      p2 = t;
    }
    if (p1 != c)
    {
      for (int i = 0; i < p.n; ++i)
        c[i] = fct != T(1) ? cpx<T>{ p1[i].r * fct, p1[i].i * fct } : p1[i];
    }
    else if (fct != T(1))
      for (int i = 0; i < p.n; ++i)
        c[i] = { c[i].r * fct, c[i].i * fct };
  }
}
