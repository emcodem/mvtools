// mvtools_vsbridge: masktools2 mt_lut / mt_lutxy / mt_adddiff computed by the bridge. See mt_port.h.
//
// Sources the semantics are taken from (both GPL 2, like this library):
//   masktools2 2.2.30 (github.com/pinterf/masktools): common/parser/symbol.cpp (evaluator, comparisons return 1 / -1,
//     == with a 1e-6 tolerance, scalef = x * max / 255 and scaleb = x << (bits - 8) for every plane of a table),
//     common/parser/symbol.h (store: clamp, (T)(v + 0.5), then min with the format's maximum),
//     filters/lut/*/ *.h (when use_expr hands the expression to Expr; realtime defaults), avs2x/filter.h (the Expr call:
//     every plane gets its expression, "x" for copy, "" for none), common/params/params.h (Y / U / V modes)
//   AviSynth+ 3.7.5 avs_core/filters/exprfilter/exprfilter.cpp: float32 evaluation, comparisons 1 / 0, scalef / scaleb
//     through get_bits_conv_constants (convert/convert_helper.h) with the chroma rule on U / V, the JIT's round
//     (roundps: half to even), store (uint)(clamp(v) + 0.5f); constant folding and the x*1, x+0 removals of
//     foldConstants do not change a value.
// Verified against masktools2 2.2.30 + AviSynth+ 3.7.5 on Windows: vsbridge/test/mt_tables.py.

#include "mt_port.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>

// The float32 evaluation must not be contracted into FMA (AviSynth+'s Expr multiplies and adds separately):
// CMakeLists.txt compiles this file with -ffp-contract=off (MSVC does not contract without /fp:fast or /fp:contract).

namespace
{

enum Code
{
  C_CONST,
  C_X,
  C_Y,
  C_ADD,
  C_SUB,
  C_MUL,
  C_DIV,
  C_MIN,
  C_MAX,
  C_ABS,
  C_SQRT,
  C_CLIP,
  C_ROUND,
  C_FLOOR,
  C_CEIL,
  C_TRUNC,
  C_LT,
  C_LE,
  C_GT,
  C_GE,
  C_EQ,
  C_NE,
  C_TERN,
  C_AND,
  C_OR,
  C_ANDNOT,
  C_XOR,
  C_NOT,
  C_DUP,
  C_SWAP
};

struct Op
{
  Code c;
  double v; // C_CONST: the value (a float value for Expr)
  int n;    // C_DUP / C_SWAP: the stack position
};

constexpr int kMaxStack = 64;

// one expression for one plane, compiled for one evaluator
struct Program
{
  bool expr = false; // AviSynth+ Expr (float32) instead of masktools' own evaluator (double)
  int bits = 8;
  std::vector<Op> ops;
};

std::string lower(std::string s)
{
  for (char &c : s)
    c = (char)tolower((unsigned char)c);
  return s;
}

// numbers as both parse them: masktools into double, Expr into float, each with a classic-locale istringstream
bool parse_number(const std::string &t, bool as_float, double &v)
{
  std::istringstream s(t);
  s.imbue(std::locale::classic());
  if (as_float)
  {
    float f;
    if (!(s >> f))
      return false;
    std::string rest;
    if (s >> rest)
      return false;
    v = f;
    return true;
  }
  double d;
  s >> d;
  if (s.fail() || !s.eof())
    return false;
  v = d;
  return true;
}

// dup / dupN, swap / swapN (N 0..9 for dup, 1..9 for swap)
bool stack_op(const std::string &t, const char *name, int def, int &n)
{
  const size_t l = strlen(name);
  if (t.compare(0, l, name) != 0)
    return false;
  if (t.size() == l)
  {
    n = def;
    return true;
  }
  if (t.size() == l + 1 && t[l] >= '0' && t[l] <= '9')
  {
    n = t[l] - '0';
    return true;
  }
  return false;
}

// AviSynth+ get_bits_conv_constants (convert_helper.h) for 8 bit -> bits, integer formats
void bits_conv(bool chroma, bool full, int bits, float &src_offset, float &mul, float &dst_offset)
{
  src_offset = 0.0f;
  mul = 1.0f;
  dst_offset = 0.0f;
  if (bits == 8)
    return;
  float src_span, dst_span;
  if (chroma)
  {
    src_offset = (float)(1 << 7);
    dst_offset = (float)(1 << (bits - 1));
    src_span = full ? (float)((1 << 8) - 1) / 2.0f : (float)(112 << 0);
    dst_span = full ? (float)((1 << bits) - 1) / 2.0f : (float)(112 << (bits - 8));
  }
  else
  {
    src_offset = full ? 0.0f : (float)(16 << 0);
    dst_offset = full ? 0.0f : (float)(16 << (bits - 8));
    src_span = full ? (float)((1 << 8) - 1) : (float)(219 << 0);
    dst_span = full ? (float)((1 << bits) - 1) : (float)(219 << (bits - 8));
  }
  mul = dst_span / src_span;
}

// compiles expr for one plane; false and err when a token is outside the port
bool compile(const std::string &expr, bool use_expr_engine, int bits, bool chroma, int nclips, Program &p,
             std::string &err)
{
  p = Program();
  p.expr = use_expr_engine;
  p.bits = bits;
  const bool E = use_expr_engine;
  std::istringstream in(expr);
  std::string t;
  int depth = 0;
  auto push = [&](Code c, double v = 0.0, int n = 0) { p.ops.push_back(Op{c, v, n}); };
  auto pushc = [&](double v) { push(C_CONST, E ? (double)(float)v : v); };
  auto need = [&](int k) {
    if (depth < k)
    {
      err = "stack underflow at '" + t + "'";
      return false;
    }
    return true;
  };
  static const std::map<std::string, Code> binary_own = {
      {"+", C_ADD}, {"-", C_SUB}, {"*", C_MUL}, {"/", C_DIV}, {"min", C_MIN}, {"max", C_MAX}, {"<", C_LT},
      {"<=", C_LE}, {">", C_GT}, {">=", C_GE}, {"==", C_EQ}, {"=", C_EQ}, {"!=", C_NE}, {"&", C_AND}, {"|", C_OR},
      {"&!", C_ANDNOT}};
  static const std::map<std::string, Code> binary_expr = {
      {"+", C_ADD}, {"-", C_SUB}, {"*", C_MUL}, {"/", C_DIV}, {"min", C_MIN}, {"max", C_MAX}, {"<", C_LT},
      {"<=", C_LE}, {">", C_GT}, {">=", C_GE}, {"==", C_EQ}, {"=", C_EQ}, {"!=", C_NE}, {"and", C_AND},
      {"&", C_AND}, {"or", C_OR}, {"|", C_OR}, {"xor", C_XOR}};
  static const std::map<std::string, Code> unary_own = {
      {"abs", C_ABS}, {"round", C_ROUND}, {"floor", C_FLOOR}, {"ceil", C_CEIL}, {"trunc", C_TRUNC}};
  static const std::map<std::string, Code> unary_expr = {{"abs", C_ABS},     {"sqrt", C_SQRT}, {"round", C_ROUND},
                                                         {"floor", C_FLOOR}, {"ceil", C_CEIL}, {"trunc", C_TRUNC},
                                                         {"not", C_NOT}};
  const auto &binary = E ? binary_expr : binary_own;
  const auto &unary = E ? unary_expr : unary_own;
  const int sh = bits - 8;
  while (in >> t)
  {
    int n;
    double v;
    if (t == "x" || (t == "y" && nclips >= 2))
    {
      push(t == "x" ? C_X : C_Y);
      ++depth;
    }
    else if (binary.count(t))
    {
      if (!need(2))
        return false;
      push(binary.at(t));
      --depth;
    }
    else if (unary.count(t))
    {
      if (!need(1))
        return false;
      push(unary.at(t));
    }
    else if (t == "?" || t == "clip")
    {
      if (!need(3))
        return false;
      push(t == "?" ? C_TERN : C_CLIP);
      depth -= 2;
    }
    else if (stack_op(t, "dup", 0, n))
    {
      if (!need(n + 1))
        return false;
      push(C_DUP, 0.0, n);
      ++depth;
    }
    else if (stack_op(t, "swap", 1, n) && n >= 1)
    {
      if (!need(n + 1))
        return false;
      push(C_SWAP, 0.0, n);
    }
    else if (t == "range_half" || t == "range_max" || t == "range_min" || t == "range_size" || t == "ymin" ||
             t == "ymax" || t == "cmin" || t == "cmax")
    {
      // integer formats: the same in both evaluators and for luma / chroma
      const double c = t == "range_half"   ? (double)(128 << sh)
                       : t == "range_max"  ? (double)((1 << bits) - 1)
                       : t == "range_min"  ? 0.0
                       : t == "range_size" ? (double)(1 << bits)
                       : t == "ymin" || t == "cmin" ? (double)(16 << sh)
                       : t == "ymax"                ? (double)(235 << sh)
                                                    : (double)(240 << sh);
      pushc(c);
      ++depth;
    }
    else if (t == "pi")
    {
      pushc(E ? (double)3.141592653589793f : 3.1415927);
      ++depth;
    }
    else if (t == "scalef" || t == "scaleb" || t == "yscalef" || t == "yscaleb")
    {
      if (!need(1))
        return false;
      const bool full = t == "scalef" || t == "yscalef";
      if (!E)
      {
        // masktools (table: chroma = false for every plane): 8-16 bit stretch x * max_t / max_s, shift x * 2^(t - s)
        if (bits != 8)
        {
          pushc(full ? (double)((1 << bits) - 1) : (double)(1 << sh));
          push(C_MUL);
          if (full)
          {
            pushc(255.0);
            push(C_DIV);
          }
        }
      }
      else
      {
        float so, mul, dof;
        bits_conv(chroma && t[0] != 'y', full, bits, so, mul, dof);
        if (so != 0.0f)
        {
          pushc(so);
          push(C_SUB);
        }
        if (mul != 1.0f)
        {
          pushc(mul);
          push(C_MUL);
        }
        if (dof != 0.0f)
        {
          pushc(dof);
          push(C_ADD);
        }
      }
    }
    else if (parse_number(t, E, v))
    {
      push(C_CONST, v);
      ++depth;
    }
    else
    {
      err = "'" + t + "' is not supported under VapourSynth (the bridge's port of masktools2 covers + - * / min max "
                      "abs clip round floor ceil trunc, comparisons, ?, " +
            std::string(E ? "and or xor not sqrt" : "& | &!") +
            ", dup / swap, range_* / ymin / ymax / cmin / cmax, pi, scalef / scaleb; pow, %, exp, log and the "
            "trigonometric functions are not: their results depend on the C library or the CPU code path)";
      return false;
    }
    if (depth > kMaxStack)
    {
      err = "expression too deep";
      return false;
    }
  }
  if (depth != 1)
  {
    err = depth == 0 ? "empty expression" : "stack unbalanced at the end of the expression";
    return false;
  }
  return true;
}

// masktools' min / max (utils.h): a < b ? a : b, a > b ? a : b
template <typename T> T mt_min(T a, T b) { return a < b ? a : b; }
template <typename T> T mt_max(T a, T b) { return a > b ? a : b; }

// masktools convert<Int64, double>: half away from zero
double mt_round(double x) { return x >= 0 ? (double)(int64_t)(x + 0.5) : (double)(int64_t)(x - 0.5); }

template <typename T> int run(const Program &p, int x, int y)
{
  T st[kMaxStack + 1];
  int sp = -1; // index of the top
  const bool E = std::is_same<T, float>::value;
  const T one = 1, no = E ? (T)0 : (T)-1;
  for (const Op &o : p.ops)
  {
    switch (o.c)
    {
    case C_CONST:
      st[++sp] = (T)o.v;
      break;
    case C_X:
      st[++sp] = (T)x;
      break;
    case C_Y:
      st[++sp] = (T)y;
      break;
    case C_DUP:
      // masktools (symbol.cpp, rec_compute) keeps the top in `last`: dupN with N >= 1 sets last to the N-th value below
      // and then pushes last, so the old top is lost ([a, b] dup1 -> [a, a, a]); Expr: [a, b] dup1 -> [a, b, a]
      if (!E && o.n > 0)
        st[sp] = st[sp - o.n];
      st[sp + 1] = st[sp - (E ? o.n : 0)];
      ++sp;
      break;
    case C_SWAP:
      std::swap(st[sp], st[sp - o.n]);
      break;
    case C_ABS:
      st[sp] = std::abs(st[sp]);
      break;
    case C_SQRT:
      st[sp] = std::sqrt(st[sp]);
      break;
    case C_ROUND:
      // Expr: the JIT's roundps (round to nearest even, the default rounding mode); masktools: half away from zero
      st[sp] = E ? std::nearbyint(st[sp]) : (T)mt_round((double)st[sp]);
      break;
    case C_FLOOR:
      st[sp] = std::floor(st[sp]);
      break;
    case C_CEIL:
      st[sp] = std::ceil(st[sp]);
      break;
    case C_TRUNC:
      st[sp] = E ? std::trunc(st[sp]) : (T)(double)(int64_t)st[sp];
      break;
    case C_NOT:
      st[sp] = st[sp] > 0 ? (T)0 : (T)1;
      break;
    case C_TERN:
      sp -= 2;
      st[sp] = st[sp] > 0 ? st[sp + 1] : st[sp + 2];
      break;
    case C_CLIP:
      sp -= 2;
      // Expr: max(min(a, high), low); masktools clip<double>(a, low, high) = min(high, max(low, a))
      st[sp] = E ? std::max(std::min(st[sp], st[sp + 2]), st[sp + 1]) : mt_min(st[sp + 2], mt_max(st[sp + 1], st[sp]));
      break;
    default:
    {
      const T b = st[sp--];
      const T a = st[sp];
      T r;
      switch (o.c)
      {
      case C_ADD: r = a + b; break;
      case C_SUB: r = a - b; break;
      case C_MUL: r = a * b; break;
      case C_DIV: r = a / b; break;
      // Expr: std::max(b, a) / std::min(b, a) (top first); masktools: min / max of (a, b)
      case C_MIN: r = E ? std::min(b, a) : mt_min(a, b); break;
      case C_MAX: r = E ? std::max(b, a) : mt_max(a, b); break;
      case C_LT: r = a < b ? one : no; break;
      case C_LE: r = a <= b ? one : no; break;
      case C_GT: r = a > b ? one : no; break;
      case C_GE: r = a >= b ? one : no; break;
      case C_EQ: r = (E ? a == b : std::abs(a - b) < (T)0.000001) ? one : no; break;
      case C_NE: r = (E ? a != b : std::abs(a - b) >= (T)0.000001) ? one : no; break;
      case C_AND: r = a > 0 && b > 0 ? one : no; break;
      case C_OR: r = a > 0 || b > 0 ? one : no; break;
      case C_ANDNOT: r = a > 0 && b <= 0 ? one : no; break;
      case C_XOR: r = (a > 0) != (b > 0) ? one : no; break;
      default: r = 0; break;
      }
      st[sp] = r;
    }
    }
  }
  const T v = st[0];
  const int maxv = (1 << p.bits) - 1;
  if (v != v) // NaN: both stores give 0 (x86 conversion of the indefinite value, Expr's max(0, NaN))
    return 0;
  if (E)
    return (int)(std::max(0.0f, std::min((float)v, (float)maxv)) + 0.5f);
  // masktools clip<Byte / Word, double>: clamp to the type, (T)(v + 0.5), then (10-14 bit) min with the maximum
  const double tmax = p.bits == 8 ? 255.0 : 65535.0;
  const int r = (int)(mt_min(tmax, mt_max(0.0, (double)v)) + 0.5);
  return std::min(r, maxv);
}

int eval(const Program &p, int x, int y) { return p.expr ? run<float>(p, x, y) : run<double>(p, x, y); }

// masktools plane modes (params.h: 1 none, 2 copy, 3 process, 4 copy second, 5 / 6 copy third / fourth, else memset)
enum Mode
{
  M_PROCESS,
  M_FIRST, // 2, and 1 (none: masktools works in place on the first clip's frame, Expr gets "" = copy the first clip)
  M_SECOND
};

enum Kind
{
  K_LUT,
  K_LUTXY,
  K_ADDDIFF
};

class MtPortClip : public IClip
{
  Kind kind;
  std::string name;
  PClip a, b;
  VideoInfo vi;
  int nplanes, bits;
  int planes[3];
  Mode mode[3];
  int prog_of[3];                      // index into progs, -1 when the plane is not processed by a table
  std::vector<Program> progs;
  std::vector<std::vector<uint16_t>> tables; // per program: lut 2^bits entries, lutxy 2^(2 bits) (bits <= 10)
  std::unique_ptr<std::once_flag[]> once;

  bool tabled(const Program &) const { return kind == K_LUT || bits <= 10; }

  const uint16_t *table(int k)
  {
    std::call_once(once[k], [&] {
      const Program &p = progs[k];
      const int n = 1 << bits;
      std::vector<uint16_t> &t = tables[k];
      if (kind == K_LUT)
      {
        t.resize(n);
        for (int x = 0; x < n; ++x)
          t[x] = (uint16_t)eval(p, x, 0);
      }
      else
      {
        t.resize((size_t)n * n);
        for (int x = 0; x < n; ++x)
          for (int y = 0; y < n; ++y)
            t[((size_t)x << bits) + y] = (uint16_t)eval(p, x, y);
      }
    });
    return tables[k].data();
  }

  template <typename P> void process(int k, const BYTE *pa, int pitch_a, const BYTE *pb, int pitch_b, BYTE *pd,
                                     int pitch_d, int w, int h)
  {
    const int maxv = (1 << bits) - 1, half = 128 << (bits - 8);
    const Program *p = prog_of[k] >= 0 ? &progs[prog_of[k]] : nullptr;
    const uint16_t *t = p && tabled(*p) ? table(prog_of[k]) : nullptr;
    for (int yy = 0; yy < h; ++yy)
    {
      const P *ra = reinterpret_cast<const P *>(pa + (size_t)yy * pitch_a);
      const P *rb = pb ? reinterpret_cast<const P *>(pb + (size_t)yy * pitch_b) : nullptr;
      P *rd = reinterpret_cast<P *>(pd + (size_t)yy * pitch_d);
      for (int xx = 0; xx < w; ++xx)
      {
        const int x = ra[xx];
        int r;
        switch (kind)
        {
        case K_LUT:
          r = t[x];
          break;
        case K_LUTXY:
          r = t ? t[((size_t)x << bits) + rb[xx]] : eval(*p, x, rb[xx]);
          break;
        default: // K_ADDDIFF: masktools adddiff, x + y - half clamped
          r = std::min(std::max(x + rb[xx] - half, 0), maxv);
          break;
        }
        rd[xx] = (P)r;
      }
    }
  }

public:
  MtPortClip(Kind k, const std::string &n, const PClip &c1, const PClip &c2, const Mode m[3],
             const std::string *exprs, bool use_expr_engine, IScriptEnvironment *env)
      : kind(k), name(n), a(c1), b(c2), vi(c1->GetVideoInfo())
  {
    bits = vi.BitsPerComponent();
    nplanes = vi.NumComponents();
    const bool rgb = vi.IsPlanarRGB();
    const int yuv[3] = {PLANAR_Y, PLANAR_U, PLANAR_V}, gbr[3] = {PLANAR_G, PLANAR_B, PLANAR_R};
    for (int i = 0; i < 3; ++i)
    {
      planes[i] = rgb ? gbr[i] : yuv[i];
      mode[i] = m[i];
      prog_of[i] = -1;
    }
    if (kind == K_LUT || kind == K_LUTXY)
    {
      std::map<std::pair<std::string, bool>, int> seen; // same expression and chroma rule: one table
      for (int i = 0; i < nplanes; ++i)
      {
        if (mode[i] != M_PROCESS)
          continue;
        const bool chroma = i > 0 && !rgb && use_expr_engine; // masktools' own tables use the luma rule everywhere
        const auto key = std::make_pair(exprs[i], chroma);
        auto it = seen.find(key);
        if (it == seen.end())
        {
          Program p;
          std::string err;
          if (!compile(exprs[i], use_expr_engine, bits, chroma, kind == K_LUT ? 1 : 2, p, err))
            env->ThrowError("%s: expression \"%s\": %s", name.c_str(), exprs[i].c_str(), err.c_str());
          it = seen.emplace(key, (int)progs.size()).first;
          progs.push_back(std::move(p));
        }
        prog_of[i] = it->second;
      }
      tables.resize(progs.size());
      once.reset(new std::once_flag[std::max<size_t>(1, progs.size())]);
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override
  {
    PVideoFrame fa = a->GetFrame(n, env);
    PVideoFrame fb = b ? b->GetFrame(n, env) : PVideoFrame();
    PVideoFrame d = env->NewVideoFrameP(vi, &fa);
    for (int i = 0; i < nplanes; ++i)
    {
      const int pl = planes[i];
      const int w = fa->GetRowSize(pl) / vi.ComponentSize(), h = fa->GetHeight(pl);
      if (mode[i] != M_PROCESS)
      {
        const PVideoFrame &s = mode[i] == M_SECOND ? fb : fa;
        env->BitBlt(d->GetWritePtr(pl), d->GetPitch(pl), s->GetReadPtr(pl), s->GetPitch(pl), s->GetRowSize(pl), h);
        continue;
      }
      const BYTE *pb = fb ? fb->GetReadPtr(pl) : nullptr;
      const int pitch_b = fb ? fb->GetPitch(pl) : 0;
      if (bits == 8)
        process<uint8_t>(i, fa->GetReadPtr(pl), fa->GetPitch(pl), pb, pitch_b, d->GetWritePtr(pl), d->GetPitch(pl), w, h);
      else
        process<uint16_t>(i, fa->GetReadPtr(pl), fa->GetPitch(pl), pb, pitch_b, d->GetWritePtr(pl), d->GetPitch(pl), w,
                          h);
    }
    return d;
  }
  bool __stdcall GetParity(int n) override { return a->GetParity(n); }
  void __stdcall GetAudio(void *, int64_t, int64_t, IScriptEnvironment *) override {}
  int __stdcall SetCacheHints(int cachehints, int) override
  {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }
  const VideoInfo &__stdcall GetVideoInfo() override { return vi; }
};

// masktools Y / U / V value -> mode; false for modes outside the port (memset, copy third / fourth)
bool plane_mode(const AVSValue &v, int def, Kind kind, Mode &m)
{
  const int i = v.Defined() ? (int)(float)v.AsFloat((float)def) : def;
  switch (i)
  {
  case 3: m = M_PROCESS; return true;
  case 2:
  case 1: m = M_FIRST; return true;
  case 4:
    m = M_SECOND;
    return kind != K_LUT;
  default: return false;
  }
}

} // namespace

PClip mt_port_invoke(const std::string &l, const AVSValue &args, const char *const *arg_names,
                     IScriptEnvironment *env)
{
  Kind kind;
  if (l == "mt_lut")
    kind = K_LUT;
  else if (l == "mt_lutxy")
    kind = K_LUTXY;
  else if (l == "mt_adddiff")
    kind = K_ADDDIFF;
  else
    return PClip();
  const char *name = l.c_str();
  const int nclips = kind == K_LUT ? 1 : 2;
  const bool lut = kind == K_LUT || kind == K_LUTXY;

  // arguments: the clips and (lut*) expr positionally, the rest by name
  std::vector<PClip> clips;
  std::map<std::string, AVSValue> named;
  const int n_in = args.IsArray() ? args.ArraySize() : 1;
  for (int i = 0; i < n_in; ++i)
  {
    const AVSValue &v = args.IsArray() ? args[i] : args;
    const char *nm = arg_names ? arg_names[i] : nullptr;
    if (nm)
      named[lower(nm)] = v;
    else if ((int)clips.size() < nclips && v.IsClip())
      clips.push_back(v.AsClip());
    else if (lut && (int)clips.size() == nclips && !named.count("expr") && v.IsString())
      named["expr"] = v;
    else
      env->ThrowError("%s: unexpected positional argument %d (not supported under VapourSynth)", name, i + 1);
  }
  if ((int)clips.size() != nclips)
    env->ThrowError("%s: needs %d clip(s)", name, nclips);

  static const char *const common[] = {"y", "u", "v", "chroma", "offx", "offy", "w", "h", "a", "alpha", "paramscale",
                                       "stacked"};
  static const char *const lut_only[] = {"expr", "yexpr", "uexpr", "vexpr", "aexpr", "realtime", "scale_inputs",
                                         "clamp_float", "clamp_float_uv", "use_expr"};
  for (const auto &kv : named)
  {
    bool ok = std::find_if(std::begin(common), std::end(common), [&](const char *s) { return kv.first == s; }) !=
              std::end(common);
    ok = ok || (lut && std::find_if(std::begin(lut_only), std::end(lut_only),
                                    [&](const char *s) { return kv.first == s; }) != std::end(lut_only));
    if (!ok)
      env->ThrowError("%s: argument '%s' is not supported under VapourSynth", name, kv.first.c_str());
  }
  auto arg = [&](const char *k) { auto it = named.find(k); return it == named.end() ? AVSValue() : it->second; };

  const VideoInfo &vi = clips[0]->GetVideoInfo();
  const int bits = vi.BitsPerComponent();
  if (!vi.IsPlanar() || vi.IsYUY2() || bits > 16 || vi.NumComponents() > 3 || vi.IsYUVA() || vi.IsPlanarRGBA())
    env->ThrowError("%s: under VapourSynth only planar 8-16 bit Y / YUV / RGB without alpha is supported", name);
  for (int k = 1; k < nclips; ++k)
  {
    const VideoInfo &v2 = clips[k]->GetVideoInfo();
    if (v2.pixel_type != vi.pixel_type || v2.width != vi.width || v2.height != vi.height)
      env->ThrowError("%s: the clips must have the same format and size", name);
  }
  if ((arg("offx").Defined() && arg("offx").AsInt() != 0) || (arg("offy").Defined() && arg("offy").AsInt() != 0) ||
      (arg("w").Defined() && arg("w").AsInt() != -1) || (arg("h").Defined() && arg("h").AsInt() != -1) ||
      (arg("stacked").Defined() && arg("stacked").AsBool()))
    env->ThrowError("%s: offx / offy / w / h / stacked are not supported under VapourSynth", name);
  if (arg("scale_inputs").Defined() && lower(arg("scale_inputs").AsString("none")) != "none")
    env->ThrowError("%s: scale_inputs is not supported under VapourSynth", name);

  // plane modes: masktools defaults Y=3, U=1, V=1; chroma="..." overrides U and V
  Mode m[3];
  const AVSValue yuv[3] = {arg("y"), arg("u"), arg("v")};
  for (int i = 0; i < 3; ++i)
    if (!plane_mode(yuv[i], i == 0 ? 3 : 1, kind, m[i]))
      env->ThrowError("%s: %c=%s is not supported under VapourSynth (only 1, 2, 3%s)", name, "YUV"[i],
                      yuv[i].IsString() ? yuv[i].AsString() : std::to_string(yuv[i].AsFloat(0)).c_str(),
                      kind == K_LUT ? "" : ", 4");
  if (arg("chroma").Defined())
  {
    const std::string c = lower(arg("chroma").AsString(""));
    Mode cm;
    if (c == "process")
      cm = M_PROCESS;
    else if (c == "copy" || c == "copy first" || c == "none" || c == "ignore")
      cm = M_FIRST;
    else if (c == "copy second" && kind != K_LUT)
      cm = M_SECOND;
    else
      env->ThrowError("%s: chroma=\"%s\" is not supported under VapourSynth", name, c.c_str());
    m[1] = m[2] = cm;
  }

  std::string exprs[3];
  bool use_expr_engine = false;
  if (lut)
  {
    const int ue = arg("use_expr").Defined() ? arg("use_expr").AsInt() : 0;
    if (ue < 0 || ue > 3)
      env->ThrowError("%s: invalid value for parameter 'use_expr'", name);
    // masktools 2.2.30: lutxy is realtime from 14 bit on unless realtime is given; lut only when asked
    const bool realtime = arg("realtime").Defined() ? arg("realtime").AsBool() : (kind == K_LUTXY && bits >= 14);
    use_expr_engine = (ue == 1 && bits > 8) || (ue == 2 && realtime) || ue == 3;
    const char *per[3] = {"yexpr", "uexpr", "vexpr"};
    for (int i = 0; i < 3; ++i)
      exprs[i] = arg(per[i]).Defined() ? arg(per[i]).AsString("x") : arg("expr").AsString("x");
  }
  return new MtPortClip(kind, name, clips[0], nclips > 1 ? clips[1] : PClip(), m, exprs, use_expr_engine, env);
}
