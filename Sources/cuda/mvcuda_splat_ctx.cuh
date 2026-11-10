// SplatFps device pipeline (one context per filter instance): source pictures are uploaded once as codes (planar YUV,
// 8 .. 16 bit, 4:4:4 / 4:2:2 / 4:2:0), turned into 4:4:4 samples (input), ME pictures A / B and phase-correlation
// spectra (three rotating slots) on the device; the fields of a picture (phase correlation to local fallback) are made
// once and kept while output frames need them; an output frame runs the pair statistic (scene cut, temporal fallback;
// sums on the device, the formula on the host), the picture build and the output codes, then one download (frame_start
// queues it, frame_finish waits for it). Included last by both variants, inside namespace SPLAT_NS.

#include "mvcuda_splat_ctx.h"

#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace SPLAT_NS
{
  static_assert(splat::CHROMA_HB_TAPS == 7, "k_s0 lists 7 half-band taps");
  constexpr float HB0 = splat::CHROMA_HB[0], HB1 = splat::CHROMA_HB[1], HB2 = splat::CHROMA_HB[2],
    HB3 = splat::CHROMA_HB[3], HB4 = splat::CHROMA_HB[4], HB5 = splat::CHROMA_HB[5], HB6 = splat::CHROMA_HB[6];
  constexpr float C420_0 = splat::CHROMA_420[0], C420_1 = splat::CHROMA_420[1], C420_2 = splat::CHROMA_420[2];

  // Code planes of one picture in device memory (Y, U, V packed: Y W x H, chroma cw x ch), T = uint8_t (8 bit) or
  // uint16_t; the format as MvCudaSplatParams gives it. The arithmetic is that of splat_core.cpp (input_rows /
  // output_rows, to_val / to_code), operation by operation.
  struct CodeFmt
  {
    int W, H, cw, ch, bits, sub_w, sub_h, centre;
  };

  // ---- input: codes -> 4:4:4 samples: value = (code - 2^(b-1)) / 2^(b-8); chroma 4:2:0 -> 4:2:2 by the 3 vertical
  // taps CHROMA_420 (MPEG-2 siting), 4:2:2 -> 4:4:4 by the 14-tap half-band (edge clamp): the even samples are the
  // input samples, the odd ones lie half way between two of them
  __device__ __forceinline__ float to_val(int code, int bits)
  {
    const float mid = (float)(1 << (bits - 1)), inv = 1.0f / (float)(1 << (bits - 8));
    return ((float)code - mid) * inv;
  }

  // chroma sample (j, y) at the 4:2:2 stage (full height): the code itself, or the vertical 4:2:0 interpolation
  template <class T> __device__ __forceinline__ float chroma422(const T *s, const CodeFmt &f, int j, int y)
  {
    if (!f.sub_h)
      return to_val(s[(size_t)y * f.cw + j], f.bits);
    const int r = y >> 1, t = (y & 1) ? min(r + 1, f.ch - 1) : max(r - 1, 0),
              a = (y & 1) ? max(r - 1, 0) : min(r + 1, f.ch - 1);
    float acc = C420_0 * to_val(s[(size_t)r * f.cw + j], f.bits);
    acc = acc + C420_1 * to_val(s[(size_t)t * f.cw + j], f.bits);
    acc = acc + C420_2 * to_val(s[(size_t)a * f.cw + j], f.bits);
    return acc;
  }

  template <class T> __global__ void k_s0(const T *codes, CodeFmt f, float *out)
  {
    const int W = f.W, H = f.H;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (x >= W || row >= H)
      return;
    const size_t NP = (size_t)W * H;
    if (c == 0)
    {
      out[(size_t)row * W + x] = to_val(codes[(size_t)row * W + x], f.bits);
      return;
    }
    const T *s = codes + NP + (size_t)(c - 1) * f.cw * f.ch;
    float r;
    if (!f.sub_w)
      r = chroma422(s, f, x, row);
    else if (f.centre)
    {
      // centre siting (splat_core.cpp chroma_upsample_row_centre): input j, the neighbour on the side of output x,
      // the one on the other side
      const int j = x >> 1, w2 = f.cw;
      const float c0 = chroma422(s, f, j, row), l = chroma422(s, f, max(j - 1, 0), row),
                  rr = chroma422(s, f, min(j + 1, w2 - 1), row);
      float a = C420_0 * c0;
      a = a + C420_1 * ((x & 1) ? rr : l);
      a = a + C420_2 * ((x & 1) ? l : rr);
      r = a;
    }
    else if (!(x & 1))
      r = chroma422(s, f, x >> 1, row);
    else
    {
      // half-band taps of splat_const.h (CHROMA_HB)
      const float HB[splat::CHROMA_HB_TAPS] = { HB0, HB1, HB2, HB3, HB4, HB5, HB6 };
      const int j = x >> 1, w2 = f.cw;
      float acc = 0.0f;
      for (int k = 0; k < splat::CHROMA_HB_TAPS; k++)
      {
        const float sum = chroma422(s, f, clampi(j - k, 0, w2 - 1), row) + chroma422(s, f, clampi(j + 1 + k, 0, w2 - 1), row);
        acc = acc + HB[k] * sum;
      }
      r = acc;
    }
    out[c * NP + (size_t)row * W + x] = r;
  }

  // ---- output: 4:4:4 samples -> codes: 2^(b-1) + round(v 2^(b-8)), clamped, round half up; chroma: even samples (left
  // siting) or the half-band pairs around 2 x + 1/2 (centre siting); 4:2:0 also the half-band pairs down the column
  __device__ __forceinline__ int to_code(float v, int bits)
  {
    const float t = v * (float)(1 << (bits - 8)) + 0.5f;
    const int c = (1 << (bits - 1)) + (int)floorf(t), top = (1 << bits) - 1;
    return c < 0 ? 0 : (c > top ? top : c);
  }

  template <class T> __global__ void k_s11(const float *in, CodeFmt f, T *codes)
  {
    const int W = f.W, H = f.H;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    const int w = c ? f.cw : W, h = c ? f.ch : H;
    if (x >= w || row >= h)
      return;
    const size_t NP = (size_t)W * H;
    if (c == 0)
    {
      codes[(size_t)row * W + x] = (T)to_code(in[(size_t)row * W + x], f.bits);
      return;
    }
    const float *s = in + c * NP;
    const int xs = x << f.sub_w;
    const float HB[splat::CHROMA_HB_TAPS] = { HB0, HB1, HB2, HB3, HB4, HB5, HB6 };
    // the 4:2:2-stage sample at full-resolution column X: the sample itself, or the vertical half-band pairs (4:2:0)
    auto col = [&](int X) {
      if (!f.sub_h)
        return s[(size_t)row * W + X];
      float acc = 0.0f;
      for (int k = 0; k < splat::CHROMA_HB_TAPS; k++)
        acc = acc + HB[k] * (s[(size_t)max(2 * row - k, 0) * W + X] + s[(size_t)min(2 * row + 1 + k, H - 1) * W + X]);
      return acc;
    };
    float v;
    if (f.sub_w && f.centre)
    {
      float acc = 0.0f;
      for (int k = 0; k < splat::CHROMA_HB_TAPS; k++)
        acc = acc + HB[k] * (col(max(xs - k, 0)) + col(min(xs + 1 + k, W - 1)));
      v = acc;
    }
    else
      v = col(xs);
    codes[NP + (size_t)(c - 1) * f.cw * f.ch + (size_t)row * f.cw + x] = (T)to_code(v, f.bits);
  }

  // Fork-join helper for the host copies of a frame (8 MB at 1080p 4:2:2 10 bit: one core copies it in about 0.8 ms,
  // several in a fraction of that): run(n, fn) calls fn(i) for every i < n on the workers and the calling thread and
  // returns when all are done.
  class CopyPool
  {
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable go_, done_;
    const std::function<void(int)> *job_ = nullptr;
    int n_ = 0, next_ = 0, left_ = 0;
    unsigned gen_ = 0;
    bool quit_ = false;

    void work()
    {
      for (;;)
      {
        int i;
        const std::function<void(int)> *f;
        {
          std::lock_guard<std::mutex> lk(m_);
          if (!job_ || next_ >= n_)
            return;
          i = next_++;
          f = job_;
        }
        (*f)(i);
        std::lock_guard<std::mutex> lk(m_);
        if (--left_ == 0)
          done_.notify_all();
      }
    }
    void loop()
    {
      unsigned seen = 0;
      for (;;)
      {
        {
          std::unique_lock<std::mutex> lk(m_);
          go_.wait(lk, [&] { return quit_ || gen_ != seen; });
          if (quit_)
            return;
          seen = gen_;
        }
        work();
      }
    }

  public:
    explicit CopyPool(int workers)
    {
      for (int i = 0; i < workers; i++)
        th_.emplace_back([this] { loop(); });
    }
    ~CopyPool()
    {
      {
        std::lock_guard<std::mutex> lk(m_);
        quit_ = true;
      }
      go_.notify_all();
      for (auto &t : th_)
        t.join();
    }
    void run(int n, const std::function<void(int)> &fn)
    {
      {
        std::lock_guard<std::mutex> lk(m_);
        job_ = &fn;
        n_ = n;
        next_ = 0;
        left_ = n;
        gen_++;
      }
      go_.notify_all();
      work();
      std::unique_lock<std::mutex> lk(m_);
      done_.wait(lk, [&] { return left_ == 0; });
      job_ = nullptr;
    }
  };

  // All work runs on one stream, so buffers reused within the context need no waits; the host waits only for the
  // temporal-fallback sums (the formula runs on the host) and for a frame's download. Host copies go through
  // page-locked staging buffers: a source upload returns once its codes are staged, and the caller can stage the next
  // source while the device builds a frame (frame_start ... source ... frame_finish). Up to NOUT frames may be started
  // before the first is finished (each has its own download buffer), so the device builds the next frame while the
  // caller copies out and hands on the current one.
  class Ctx : public SplatCtx
  {
    struct Src
    {
      float *pic = nullptr, *A = nullptr, *B = nullptr; // 3 planes W x H; ME pictures
    };
    struct Fld
    {
      float *fwd = nullptr, *bwd = nullptr; // 4 planes each
    };
    static const int NSTAGE = 3, NOUT = 2;

    MvCudaSplatParams p_;
    splat::Config cfg_;
    Bias costs_;    // costs of the phase-correlation set
    int nc_ = 0;    // phase-correlation candidates per pair
    int ncap_ = 0;  // candidates the candidate buffers (vf_, vb_, F_) hold
    std::unique_ptr<Bufs> CB_; // candidate buffers (grown by candidates() when an external set is larger)
    int W, H, w, h;
    size_t NP, np, code_bytes;
    CodeFmt cf_;          // the code planes of a picture (source and output)
    size_t row_bytes_[3]; // bytes per row of the Y, U, V code planes
    int rows_[3];         // rows of the Y, U, V code planes
    cudaStream_t st_ = nullptr;
    Bufs B_; // persistent scratch
    FieldsTmp F_;
    BuildScratch BS_;
    unsigned char *codes_ = nullptr; // Y, U, V codes of one picture (in or out), packed as cf_ says
    float *tmp_ = nullptr, *vf_ = nullptr, *vb_ = nullptr, *dg_ = nullptr, *dout_ = nullptr;
    // vf_, vb_ and F_ for n candidates (waits for the stream when they exist: the old ones may be in use)
    bool alloc_cands(int n)
    {
      if (n <= ncap_)
        return true;
      if (CB_)
        cudaStreamSynchronize(st_);
      CB_.reset(new Bufs);
      F_ = FieldsTmp();
      const size_t npl = (size_t)w * h;
      vf_ = CB_->get(2 * npl * n);
      vb_ = CB_->get(2 * npl * n);
      if (CB_->failed || !F_.alloc(*CB_, w, h, n))
      {
        ncap_ = 0;
        return false;
      }
      ncap_ = n;
      return true;
    }
    double *tfr_ = nullptr;
    splat_exact::PcorrState *S_ = nullptr;
    int slot_q_[3] = { -1, -1, -1 };
    std::map<int, Src> src_;
    std::map<int, Fld> fld_;
    std::vector<Src> spare_src_;
    std::vector<Fld> spare_fld_;
    // page-locked host buffers: source staging ring, frame download, temporal-fallback sums
    unsigned char *stage_[NSTAGE] = {};
    cudaEvent_t stage_ev_[NSTAGE] = {};
    int stage_next_ = 0;
    unsigned char *pout_[NOUT] = {};
    double *ptf_ = nullptr;
    cudaEvent_t out_ev_[NOUT] = {}, tf_ev_ = nullptr;
    // started frames, oldest first: download slot (pend_head_ + i) % NOUT holds the frame of source pair p = pend_p_[]
    int pend_p_[NOUT] = {}, pend_head_ = 0, pend_n_ = 0;
    CopyPool copy_pool_{ 4 };
    // copies the 3 code planes between packed (rows of row_bytes_, planes one after the other) and pitched host memory,
    // in pieces of rows over the copy pool; to_packed: planes -> packed
    void copy_planes(unsigned char *packed, unsigned char *const planes[3], const int pitch[3], bool to_packed)
    {
      const int PIECES = 8; // per plane
      size_t base[3];
      base[0] = 0;
      for (int c = 1; c < 3; c++)
        base[c] = base[c - 1] + row_bytes_[c - 1] * rows_[c - 1];
      copy_pool_.run(3 * PIECES, [&](int i) {
        const int c = i / PIECES, k = i % PIECES, y0 = rows_[c] * k / PIECES, y1 = rows_[c] * (k + 1) / PIECES;
        for (int y = y0; y < y1; y++)
        {
          unsigned char *pk = packed + base[c] + (size_t)y * row_bytes_[c], *pl = planes[c] + (size_t)y * pitch[c];
          if (to_packed)
            memcpy(pk, pl, row_bytes_[c]);
          else
            memcpy(pl, pk, row_bytes_[c]);
        }
      });
    }
    bool external_ = false; // fields only from candidates()
    bool ok_ = false;

    void free_src(Src &s)
    {
      mvc::dev_free(s.pic, sizeof(float) * 3 * NP);
      mvc::dev_free(s.A, sizeof(float) * np);
      mvc::dev_free(s.B, sizeof(float) * np);
    }
    void free_fld(Fld &f)
    {
      mvc::dev_free(f.fwd, sizeof(float) * 4 * np);
      mvc::dev_free(f.bwd, sizeof(float) * 4 * np);
    }
    // keeps the entries near q (a frame needs the pictures p - 1 .. p + 2; output order is mostly increasing); the
    // others go to a spare list (stream order makes their reuse safe)
    template <class M, class V> static void trim(M &m, int q, V &spare)
    {
      for (auto it = m.begin(); it != m.end();)
        if (it->first < q - 4 || it->first > q + 4)
        {
          spare.push_back(it->second);
          it = m.erase(it);
        }
        else
          ++it;
    }
    static int sync(cudaStream_t st)
    {
      return (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess) ? -2 : 0;
    }
    static int wait(cudaEvent_t e)
    {
      return (cudaEventSynchronize(e) != cudaSuccess || cudaGetLastError() != cudaSuccess) ? -2 : 0;
    }
    // the spectra of source q in its slot (q mod 3)
    void slot(int q)
    {
      const int s = q % 3;
      if (slot_q_[s] != q)
      {
        splat_exact::pcorr_ingest(S_, st_, src_[q].B, s);
        slot_q_[s] = q;
      }
    }
    bool can_fields(int c) const
    {
      return c >= 0 && (c == 0 || src_.count(c - 1)) && src_.count(c) && src_.count(c + 1);
    }
    // forward and backward field of centre picture c (once): phase-correlation candidates of the pairs (c, c+1) and
    // (c-1, c) from the stored spectra, then refinement to local fallback (fields_dev); picture 0 has the forward
    // side only
    // field buffers: a spare one, else new from the pool (false: out of device memory)
    bool take_fld(Fld &f)
    {
      if (!spare_fld_.empty())
      {
        f = spare_fld_.back();
        spare_fld_.pop_back();
        return true;
      }
      f.fwd = (float *)mvc::dev_alloc(sizeof(float) * 4 * np);
      f.bwd = (float *)mvc::dev_alloc(sizeof(float) * 4 * np);
      if (f.fwd && f.bwd)
        return true;
      free_fld(f);
      return false;
    }
    int fields(int c)
    {
      if (fld_.count(c))
        return 0;
      if (external_ || !can_fields(c))
        return -4; // external candidates: candidates() makes the fields
      const bool single = c == 0;
      slot(c);
      slot(c + 1);
      splat_exact::pcorr_pair(S_, st_, c % 3, (c + 1) % 3, vf_, dg_);
      if (!single)
      {
        slot(c - 1);
        splat_exact::pcorr_pair(S_, st_, (c - 1) % 3, c % 3, vb_, dg_);
      }
      Fld f;
      if (!take_fld(f))
        return -2;
      fields_dev(st_, cfg_, F_, single ? nullptr : src_[c - 1].A, src_[c].A, src_[c + 1].A, w, h, vf_, vb_, nc_, false, true,
        costs_, f.fwd, f.bwd);
      fld_[c] = f;
      return 0;
    }

  public:
    explicit Ctx(const MvCudaSplatParams &p) : p_(p), cfg_(cfg_of(p))
    {
      p_.config = nullptr; // the caller's Config may go away; cfg_ is the copy
      W = p.width;
      H = p.height;
      me_dims(cfg_, W, H, w, h);
      nc_ = splat::pcorr_ncand(cfg_, w, h);
      costs_ = pcorr_bias(cfg_, w, h);
      NP = (size_t)W * H;
      np = (size_t)w * h;
      cf_ = { W, H, W >> p.sub_w, H >> p.sub_h, p.bits, p.sub_w, p.sub_h, p.chroma_centre };
      const size_t bps = p.bits == 8 ? 1 : 2;
      row_bytes_[0] = bps * W;
      row_bytes_[1] = row_bytes_[2] = bps * cf_.cw;
      rows_[0] = H;
      rows_[1] = rows_[2] = cf_.ch;
      code_bytes = bps * (NP + 2 * (size_t)cf_.cw * cf_.ch);
      if (cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking) != cudaSuccess)
        return;
      codes_ = (unsigned char *)B_.get((code_bytes + 3) / 4);
      tmp_ = B_.get((size_t)w * H);
      dg_ = B_.get(8);
      dout_ = B_.get(3 * NP);
      tfr_ = (double *)B_.get(2 * 2 * (size_t)h);
      if (B_.failed || !alloc_cands(nc_) || BS_.alloc(cfg_, W, H, st_))
        return;
      for (int i = 0; i < NSTAGE; i++)
        if (!(stage_[i] = (unsigned char *)mvc::pin_alloc(code_bytes)) ||
            cudaEventCreateWithFlags(&stage_ev_[i], cudaEventDisableTiming) != cudaSuccess)
          return;
      for (int i = 0; i < NOUT; i++)
        if (!(pout_[i] = (unsigned char *)mvc::pin_alloc(code_bytes)) ||
            cudaEventCreateWithFlags(&out_ev_[i], cudaEventDisableTiming) != cudaSuccess)
          return;
      ptf_ = (double *)mvc::pin_alloc(sizeof(double) * 2 * (size_t)h);
      if (!ptf_ || cudaEventCreateWithFlags(&tf_ev_, cudaEventDisableTiming) != cudaSuccess)
        return;
      S_ = splat_exact::pcorr_state_new(st_, w, h, cfg_);
      ok_ = S_ != nullptr && sync(st_) == 0;
    }

    ~Ctx() override
    {
      if (st_)
        cudaStreamSynchronize(st_);
      for (auto &s : src_)
        free_src(s.second);
      for (auto &s : spare_src_)
        free_src(s);
      for (auto &f : fld_)
        free_fld(f.second);
      for (auto &f : spare_fld_)
        free_fld(f);
      for (int i = 0; i < NSTAGE; i++)
      {
        mvc::pin_free(stage_[i], code_bytes);
        if (stage_ev_[i])
          cudaEventDestroy(stage_ev_[i]);
      }
      for (int i = 0; i < NOUT; i++)
      {
        mvc::pin_free(pout_[i], code_bytes);
        if (out_ev_[i])
          cudaEventDestroy(out_ev_[i]);
      }
      mvc::pin_free(ptf_, sizeof(double) * 2 * (size_t)h);
      if (tf_ev_)
        cudaEventDestroy(tf_ev_);
      if (S_)
        splat_exact::pcorr_state_free(S_);
      if (st_)
        cudaStreamDestroy(st_);
    }

    bool ok() const override { return ok_; }
    bool has_source(int q) const override { return src_.count(q) != 0; }
    void set_external() override { external_ = true; }
    bool has_fields(int c) const override { return fld_.count(c) != 0; }

    int candidates(int c, const float *fwd, int fwd_pitch_bytes, const float *bwd, int bwd_pitch_bytes, int n,
      bool bwd_toward, bool refine, const float *bias) override
    {
      if (fld_.count(c))
        return 0;
      if (!fwd || n < 1 || n > MAX_NC || !src_.count(c) || !src_.count(c + 1) ||
          (bwd && (c == 0 || !src_.count(c - 1))))
        return -4;
      if (!alloc_cands(n))
        return -2;
      // the sets go through the candidate buffers (stream order: the previous user of vf_ / vb_ is queued before)
      const size_t row = sizeof(float) * w;
      cudaMemcpy2DAsync(vf_, row, fwd, fwd_pitch_bytes, row, (size_t)2 * n * h, cudaMemcpyHostToDevice, st_);
      if (bwd)
        cudaMemcpy2DAsync(vb_, row, bwd, bwd_pitch_bytes, row, (size_t)2 * n * h, cudaMemcpyHostToDevice, st_);
      Fld f;
      if (!take_fld(f))
        return -2;
      fields_dev(st_, cfg_, F_, bwd ? src_[c - 1].A : nullptr, src_[c].A, src_[c + 1].A, w, h, vf_, vb_, n, bwd_toward,
        refine, bias_of(bias, n + 1), f.fwd, f.bwd);
      fld_[c] = f;
      return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
    }

    int source(int q, const void *const planes[3], const int pitch_bytes[3]) override
    {
      if (src_.count(q))
        return 0;
      Src s;
      if (!spare_src_.empty())
      {
        s = spare_src_.back();
        spare_src_.pop_back();
      }
      else
      {
        s.pic = (float *)mvc::dev_alloc(sizeof(float) * 3 * NP);
        s.A = (float *)mvc::dev_alloc(sizeof(float) * np);
        s.B = (float *)mvc::dev_alloc(sizeof(float) * np);
        if (!s.pic || !s.A || !s.B)
        {
          free_src(s);
          return -2;
        }
      }
      // stage the codes (Y, U, V packed as in codes_), once the slot's previous upload has left it
      const int k = stage_next_;
      stage_next_ = (k + 1) % NSTAGE;
      if (wait(stage_ev_[k]))
      {
        spare_src_.push_back(s);
        return -2;
      }
      unsigned char *in[3];
      for (int c = 0; c < 3; c++)
        in[c] = const_cast<unsigned char *>(static_cast<const unsigned char *>(planes[c])); // read only (to_packed)
      copy_planes(stage_[k], in, pitch_bytes, true);
      cudaMemcpyAsync(codes_, stage_[k], code_bytes, cudaMemcpyHostToDevice, st_);
      cudaEventRecord(stage_ev_[k], st_);
      const dim3 blk(32, 8);
      if (cf_.bits == 8)
        k_s0<<<grid(W, H, 3, blk), blk, 0, st_>>>(codes_, cf_, s.pic);
      else
        k_s0<<<grid(W, H, 3, blk), blk, 0, st_>>>((const uint16_t *)codes_, cf_, s.pic);
      me_pics_dev(st_, cfg_, s.pic, W, H, w, h, tmp_, s.A, s.B);
      trim(src_, q, spare_src_);
      src_[q] = s;
      // the fields of the previous picture as soon as its neighbours are here (the next frames need them)
      if (!external_ && can_fields(q - 1))
      {
        const int rc = fields(q - 1);
        if (rc)
          return rc;
      }
      return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
    }

    // fields of P = p and N = p + 1, the pair statistic (sums on the device, formula on the host: the one host wait per
    // frame), the build with the adjusted phase, the output codes and the download into page-locked memory
    // (frame_finish copies it out). With NOUT frames started already, the oldest is dropped.
    int frame_start(int p, double d, double strength, double *d_adj) override
    {
      if (pend_n_ == NOUT)
      {
        const int rc = frame_finish(nullptr, nullptr);
        if (rc)
          return rc;
      }
      int rc = fields(p);
      if (!rc)
        rc = fields(p + 1);
      if (rc)
        return rc;
      const Fld &fp = fld_[p], &fn = fld_[p + 1];
      // error statistic and temporal fallback: row sums on the device, the formula on the host
      tf_launch(st_, cfg_, fp.fwd, fn.bwd, w, h, tfr_);
      cudaMemcpyAsync(ptf_, tfr_, sizeof(double) * 2 * h, cudaMemcpyDeviceToHost, st_);
      cudaEventRecord(tf_ev_, st_);
      if (wait(tf_ev_))
        return -2;
      const double *hr = ptf_;
      const double dd = splat::phase_pull_from_rows(pull_sums(cfg_, w, h, hr), d, strength, cfg_, nullptr);
      if (d_adj)
        *d_adj = dd;
      rc = build_run(st_, BS_, src_[p].pic, src_[p + 1].pic, fp.fwd, fn.bwd, (float)dd, dout_);
      if (rc)
        return rc;
      const dim3 blk(32, 8);
      if (cf_.bits == 8)
        k_s11<<<grid(W, H, 3, blk), blk, 0, st_>>>(dout_, cf_, codes_);
      else
        k_s11<<<grid(W, H, 3, blk), blk, 0, st_>>>(dout_, cf_, (uint16_t *)codes_);
      const int k = (pend_head_ + pend_n_) % NOUT;
      cudaMemcpyAsync(pout_[k], codes_, code_bytes, cudaMemcpyDeviceToHost, st_);
      cudaEventRecord(out_ev_[k], st_);
      if (cudaPeekAtLastError() != cudaSuccess)
        return -2;
      pend_p_[k] = p;
      pend_n_++;
      return 0;
    }

    // the oldest started frame into out (null: dropped)
    int frame_finish(void *const out[3], const int out_pitch_bytes[3]) override
    {
      if (!pend_n_)
        return -4;
      const int k = pend_head_, p = pend_p_[k];
      pend_head_ = (k + 1) % NOUT;
      pend_n_--;
      if (wait(out_ev_[k]))
        return -2;
      if (out)
      {
        unsigned char *o[3];
        for (int c = 0; c < 3; c++)
          o[c] = static_cast<unsigned char *>(out[c]);
        copy_planes(pout_[k], o, out_pitch_bytes, false);
      }
      trim(fld_, p, spare_fld_);
      return 0;
    }
  };

  SplatCtx *ctx_new(const MvCudaSplatParams &p)
  {
    Ctx *c = new Ctx(p);
    if (!c->ok())
    {
      delete c;
      return nullptr;
    }
    return c;
  }
}
