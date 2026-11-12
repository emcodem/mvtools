# uc_FRC chain (compare/nvofa_frc/scripts/avs/frc_flow.avs) with the NVENC ME-only prototype in place of MAnalyse.
# The prototype (NVENCMEProto, scratchpad build, not in the repo) gives one-level vectors; MRecalculate refines them
# and recomputes the SADs, MFlowFps interpolates, both as in the normal chain. Settings come from the caller:
#   NVME_PLUGIN (path of nvencme_proto.dll), NVME_PRESET (1..7), NVME_HALF (true/false), NVME_LEVELS (1..4, hinted
#   NVENC pyramid), NVME_DX12 (0: NVENC SDK, 8 / 16: D3D12 ME block size), set in a generated header.
LoadPlugin("C:\path\to\plugins\LSMASHSource.dll")
LoadPlugin("C:\path\to\plugins\RgTools.dll")
LoadPlugin("C:\path\to\build_nvofa\mvtools2.dll")
LoadPlugin(NVME_PLUGIN)
SEGFILE = "C:\path\to\clips\seg1_wires_pole.mp4"
m = LWLibavVideoSource(SEGFILE, threads=4, cachefile=SEGFILE + ".lwi")
pre = m.RemoveGrain(22)
sup = MSuper(m, pel=2, hpad=16, vpad=16, rfilter=4, levels=1, gpu=true)
bv = NVENCMEProto(pre, isb=true, blksize=32, overlap=16, pel=2, hpad=16, vpad=16, preset=NVME_PRESET, half=NVME_HALF, levels=NVME_LEVELS, dx12=NVME_DX12)
fv = NVENCMEProto(pre, isb=false, blksize=32, overlap=16, pel=2, hpad=16, vpad=16, preset=NVME_PRESET, half=NVME_HALF, levels=NVME_LEVELS, dx12=NVME_DX12)
bv = MRecalculate(sup, bv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
fv = MRecalculate(sup, fv, blksize=16, overlap=8, thSAD=100, search=3, searchparam=2, truemotion=true, chroma=true, gpu=true)
out = MFlowFps(m, sup, bv, fv, num=24000, den=1001, mask=2, ml=200, blend=false, mt=true, gpu=true)
return out.Trim(8, 247).Prefetch(3)
