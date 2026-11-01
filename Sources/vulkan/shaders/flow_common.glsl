// Helpers of the MFlowFps kernels (port of Sources/cuda/mvcuda_flow.cu): SimpleResize control values, bilinear values of
// the small arrays, vector limiting. Include after common.glsl.

int sext16(int v) { return bitfieldExtract(v, 0, 16); }

struct UpPos
{
  int row1, row2, vw2;   // source rows and the weight of the second
  int hoff, hw1, hw2;    // source column and weights
  int old_w;
};

// g: new_w, new_h, old_w, old_h
void h_ctl(ivec4 g, int x, out int offs, out int w1, out int w2)
{
  int j = x * 256 * g.z / g.x + (128 * g.z) / g.x - 128;
  if (j < 0)
    j = 0;
  const int k = j >> 8;
  w2 = j - (k << 8);
  w1 = 256 - w2;
  if (k > g.z - 2)
  {
    offs = g.z - 1;
    w1 = 256;
    w2 = 0;
  }
  else
    offs = k;
}

void v_one(ivec4 g, int yy, out int o, out int w)
{
  int j = yy * 256 * g.w / g.y + (128 * g.w) / g.y - 128;
  if (j < 0)
    j = 0;
  int k = j >> 8;
  if (k > g.w - 2)
  {
    k = g.w - 1;
    j = k << 8;
  }
  o = k;
  w = j - (k << 8);
}

UpPos up_pos(ivec4 g, int x, int y)
{
  UpPos p;
  int voff, last, dummy;
  v_one(g, y, voff, p.vw2);
  v_one(g, g.y - 1, last, dummy);
  p.row1 = voff;
  p.row2 = (voff < last) ? voff + 1 : 0;
  h_ctl(g, x, p.hoff, p.hw1, p.hw2);
  p.old_w = g.z;
  return p;
}

// bilinear value of a small array of shorts (vertical pass stored as short, then horizontal)
int up_value_s(uint64_t s, UpPos p)
{
  const int inv = 256 - p.vw2;
  const int c0 = p.hoff, c1 = min(p.hoff + 1, p.old_w - 1);
  const int r1 = p.row1 * p.old_w, r2 = p.row2 * p.old_w;
  const int a = sext16((int(PI16(s).v[r1 + c0]) * inv + int(PI16(s).v[r2 + c0]) * p.vw2 + 128) >> 8);
  const int b = sext16((int(PI16(s).v[r1 + c1]) * inv + int(PI16(s).v[r2 + c1]) * p.vw2 + 128) >> 8);
  return (a * p.hw1 + b * p.hw2 + 128) >> 8;
}

// the same for a small array of bytes
int up_value_b(uint64_t s, UpPos p)
{
  const int inv = 256 - p.vw2;
  const int c0 = p.hoff, c1 = min(p.hoff + 1, p.old_w - 1);
  const int r1 = p.row1 * p.old_w, r2 = p.row2 * p.old_w;
  const int a = ((int(uint(PU8(s).v[r1 + c0])) * inv + int(uint(PU8(s).v[r2 + c0])) * p.vw2 + 128) >> 8) & 255;
  const int b = ((int(uint(PU8(s).v[r1 + c1])) * inv + int(uint(PU8(s).v[r2 + c1])) * p.vw2 + 128) >> 8) & 255;
  return (a * p.hw1 + b * p.hw2 + 128) >> 8;
}

// MakeVectorsSafe: the relative position must stay inside the frame
int limit_x(int v, int x, int real_w, int lp)
{
  if (x >= real_w)
    return sext16(v);
  const int maxr = ((real_w - x) << lp) - 1, minr = -(x << lp);
  return max(min(sext16(v), sext16(maxr)), sext16(minr));
}

int limit_y(int v, int y, int real_h, int lp)
{
  if (y >= real_h)
    return sext16(v);
  const int maxr = ((real_h - y) << lp) - 1, minr = -(y << lp);
  return max(min(sext16(v), sext16(maxr)), sext16(minr));
}
