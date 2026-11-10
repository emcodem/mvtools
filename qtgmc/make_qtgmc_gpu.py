"""Makes QTGMC_gpu.avsi from QTGMC.avsi 3.393s (realfinder AVS-Stuff, Community branch, `avs 2.6 and up/QTGMC.avsi`,
https://github.com/realfinder/AVS-Stuff): the motion analysis (MSuper of the search clip, MAnalyse,
MRecalculate) goes through wrappers that call the Cuda* filters of mvtools2 with QTGMC_gpu, the CPU filters without.

python -I make_qtgmc_gpu.py [source.avsi] [dest.avsi]

The source is read only. The global QTGMC_gpu (default false) is declared at the top of the copy; set
`global QTGMC_gpu = true` after Import to switch the GPU on. A Cuda* filter that cannot run on the GPU stops with an
error unless `global cuda_cpu_fallback = true` (then it runs on the CPU). The other mvtools calls of QTGMC are
untouched; the GPU output side (QTGMC_gpu_out) replaces them as a whole.

Every pattern must match as often as expected, else the script stops: a pattern that misses silently would leave that
part on the CPU. The line ends of the source (CRLF or LF) are kept.
"""
import re
import sys

SRC = r'C:\dev\mvtools\profile\qtgmc\QTGMC_3.393.avsi'
DST = r'C:\dev\mvtools\profile\qtgmc\QTGMC_gpu.avsi'


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else DST
    raw = open(src, 'rb').read()
    text = raw.decode('latin-1')
    crlf = '\r\n' in text
    text = text.replace('\r\n', '\n')
    n = {}

    def sub(name, pattern, repl, expect, flags=0):
        nonlocal text
        text, k = re.subn(pattern, repl, text, flags=flags)
        n[name] = k
        if k != expect:
            sys.exit('pattern %s: %d replacements, expected %d (QTGMC version changed?)' % (name, k, expect))

    # the analysis calls go through QTGMC_MSuper / QTGMC_MAnalyse / QTGMC_MRecalculate (defined at the end): CudaMSuper /
    # CudaMAnalyse / CudaMRecalculate with QTGMC_gpu, else the CPU filters; same arguments
    # (MAnalyse with the GPU: meander=QTGMC_gpu_meander; false = wavefront search, true = the slice kernel = stock result)
    # 3.39x: the calls are on srchClip_us / srchSuper_us (MEus) and MAnalyse / MRecalculate sit inside eval strings, which
    # run in QTGMC's scope; 1 MSuper line with 2 calls, 12 MAnalyse, 24 MRecalculate (RefineMotion, RefineMotion2) + 2
    # for ShutterBlur
    sub('msuper', r'\b(srchClip(?:_us)?)\.MSuper\(', r'\1.QTGMC_MSuper(', 2)
    sub('manalyse', r'\b(srchSuper(?:_us)?)\.MAnalyse\(', r'\1.QTGMC_MAnalyse(', 12)
    sub('mrecalc', r'\b(srchSuper(?:_us)?)\.MRecalculate\(', r'\1.QTGMC_MRecalculate(', 26)

    # GPU search clip (cudafilters.dll): bob, temporal soften, merges,
    # KeepOnlyBobShimmerFixes, RemoveGrain 12, Gauss / Bilinear resize, the SrchClipPP 3 mt_lutxy pair as Cuda* filters,
    # bit-exact; for every preset's pre-filter (TR0 0..2, SrchClipPP 0..3, with or without chroma motion), 4:2:2 / 4:4:4 /
    # Y input, 4:2:0 without chroma motion (with it, Slower / Very Slow / Placebo differ from the CPU chain; cause not found
    # yet), 8 to 16 bit (stages without a GPU path at a bit depth fall back to their CPU
    # filters inside). Not with PhosphorSoft (reduceflicker + Overlay) or xflateMaTo with Rep0 (mt_inflate / mt_deflate
    # in the repair); with MEus the search clip is downloaded (the upsizing LanczosResize runs on the CPU)
    sub('srchclip_gpu', r'(\n\tsrchClip    = srchcheck         \? srchClip : \\\n(?:[^\n]*\\\n)*[^\n]*\n)',
        lambda m: m.group(1) + '\tsrchClip    = (QTGMC_gpu && QTGMC_gpu_search && !srchcheck && InputType == 0 && UseEdiExt == 0 && TR0 >= 0 && TR0 <= 2 \\\n'
        '\t               && SrchClipPP >= 0 && SrchClipPP <= 3 && !yuy2 && !Border && !tv_range \\\n'
        '\t               && !(Is420(planarClip) && ChromaMotion) && !CRTSoft && !(xflateMaTo && Rep0 > 0)) \\\n'
        '\t               ? QTGMC_gpu_srchclip( planarClip, TR0, Rep0, SrchClipPP, ChromaMotion, RepChroma, w, h, epsilon, \\\n'
        '\t               !QTGMC_gpu_tokens || MEus || ProgSADMask > 0.0 || (ShutterBlur > 0 && SBlurLimit > 0) || (sbpc > 8 && FastMA) ) : srchClip\n', 1)
    # NNEDI3 interpolation (both calls) through QTGMC_gpu_nnedi3: CudaNNEDI3 (bit-exact port of plugins_JPSDR nnedi3, falls
    # back to the plugin itself where it is not exact) with QTGMC_gpu_edi
    sub('nnedi3', r'Input\.NNEDI3\(', 'Input.QTGMC_gpu_nnedi3(', 2)

    # GPU output side: with QTGMC_gpu_out, for the structures it reproduces
    # (every preset's defaults: TR1 0..2, TR2 0..3, SMode 0..2 incl. Precise, SLMode 0 / 2 with SLRad 1 / 3, Sbb 0..3,
    # SubPel 1 / 2, grain / noise restore of a CPU-made noise clip), edi -> MSuper -> MCompensate -> MDegrain -> resharp ->
    # limit -> back blend -> MDegrain -> KeepOnlyBobShimmerFixes -> noise runs on the GPU; addNoise2 is that result,
    # downloaded. The CPU definitions in between are still built but never asked for frames. An interpolation other than
    # the GPU NNEDI3 is made on the CPU and uploaded.
    sub('gpu_out_flag', r'(\n\tedi1 = defined\(EdiExt\))',
        lambda m: '\n\tQTGMC_gpuOut = QTGMC_gpu && QTGMC_gpu_out && !yuy2 && InputType == 0 && !defined(EdiExt) && ChromaEdi == "" \\\n'
        '\t               && TR1 <= 2 && TR2 <= 3 && Rep1 == 0 && SourceMatch == 0 && Lossless == 0 && SMode <= 2 && SVThin <= 0.0 && Sbb <= 3 \\\n'
        '\t               && (SLMode == 0 || SLMode == 2) && !lsb && !n16 && SubPel <= 2 && SubPelInterp == 2 && ProgSADMask <= 0.0 \\\n'
        '\t               && !(xflateMaTo && Rep2 > 0)\n'
        '\tglobal QTGMC_gpu_edi_token = QTGMC_gpuOut && QTGMC_gpu_edi && EdiMode == "NNEDI3"' + m.group(1), 1)
    sub('gpu_out_addnoise2', r'(\n\taddNoise2 = \(NoiseRestore <= 0\.0\) \? lossed2 : \\\n[^\n]*\n)',
        lambda m: m.group(1) + '\tglobal QTGMC_gpu_edi_token = false\n'
        '\taddNoise2 = QTGMC_gpuOut ? QTGMC_gpu_outside( edi, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, SubPel, SubPelInterp, hpad, vpad, \\\n'
        '\t                                              ThSCD1, ThSCD2, ThSAD1, ThSAD2, TR1, TR2, SMode, Precise, SLMode, SLRad, Sbb, \\\n'
        '\t                                              "x x y - "+ string(sharpAdj) + " * +", rgBlur, Sovs, Rep2, RepChroma, finalNoise, \\\n'
        '\t                                              "x " + noiseCentre + " scalef - " + string(GrainRestore) + " * range_half +", \\\n'
        '\t                                              "x " + noiseCentre + " scalef - " + string(NoiseRestore) + " * range_half +", \\\n'
        '\t                                              GrainRestore, NoiseRestore, CNmt1, CNmt2, w+epsilon, h+epsilon ).CudaDownload() : addNoise2\n', 1)
    text = ('global QTGMC_gpu = false # set to true after Import to run the mvtools motion analysis on the GPU (profile\\qtgmc\\make_qtgmc_gpu.py)\n'
            'global QTGMC_gpu_meander = true # with QTGMC_gpu: true = as stock QTGMC (slice search, byte-identical to the CPU-mt result), false = wavefront search (meander=false, single slice: faster on the GPU, slightly different vectors)\n'
            'global QTGMC_gpu_search = false # with QTGMC_gpu: true = the search clip is made by cudafilters.dll (needs LoadPlugin of it)\n'
            'global QTGMC_gpu_tokens = true # with QTGMC_gpu_search: the search clip stays on the device, MSuper / MAnalyse read it there\n'
            'global QTGMC_gpu_edi = false # with QTGMC_gpu: NNEDI3 by cudafilters.dll CudaNNEDI3 (needs LoadPlugin of it)\n'
            'global QTGMC_gpu_out = false # with QTGMC_gpu: the output side (MCompensate, MDegrain, resharp, limits, back blend, repair, noise restore) on the GPU\n'
            'global QTGMC_gpu_edi_token = false # internal: the interpolated clip stays on the device (set by QTGMC for the GPU output side)\n'
            + text)
    # the analysis wrappers: every parameter of the mvtools2 filter, passed on by name (an argument the caller leaves out
    # stays undefined and the filter takes its default)
    sup_p = ['hpad i', 'vpad i', 'pel i', 'levels i', 'chroma b', 'sharp i', 'rfilter i', 'pelclip c', 'isse b', 'planar b', 'mt b']
    ana_p = ['blksize i', 'blksizeV i', 'levels i', 'search i', 'searchparam i', 'pelsearch i', 'isb b', 'lambda i', 'chroma b',
             'delta i', 'truemotion b', 'lsad i', 'plevel i', 'global b', 'pnew i', 'pzero i', 'pglobal i', 'overlap i', 'overlapV i',
             'outfile s', 'dct i', 'divide i', 'sadx264 i', 'badSAD i', 'badrange i', 'isse b', 'meander b', 'temporal b', 'trymany b',
             'multi b', 'mt b', 'scaleCSAD i', 'search_coarse i', 'trymanyall b']
    rec_p = ['thsad i', 'smooth i', 'blksize i', 'blksizeV i', 'search i', 'searchparam i', 'lambda i', 'chroma b', 'truemotion b',
             'pnew i', 'overlap i', 'overlapV i', 'outfile s', 'dct i', 'divide i', 'sadx264 i', 'isse b', 'meander b', 'tr i', 'mt b',
             'scaleCSAD i']
    types = {'i': 'int', 'b': 'bool', 's': 'string', 'c': 'clip'}

    def wrapper(name, clips, params, gpu_extra=None):
        decl = ', '.join('%s "%s"' % (types[p.split()[1]], p.split()[0]) for p in params)
        def call(fn, extra):
            args = ', '.join('%s=%s' % (p.split()[0], extra.get(p.split()[0], p.split()[0])) for p in params)
            return 'c.%s( %s%s )' % (fn, ''.join(x + ', ' for x in clips[1:]), args)
        cdecl = ', '.join(['clip c'] + ['clip ' + x for x in clips[1:]])
        return ('function QTGMC_%s( %s, %s )\n{\n\treturn QTGMC_gpu ? %s \\\n\t                 : %s\n}\n'
                % (name, cdecl, decl, call('Cuda' + name, gpu_extra or {}), call(name, {})))
    text += ('\n\n# The motion analysis of QTGMC: the Cuda* filters of mvtools2 with QTGMC_gpu, else the CPU filters (same arguments).\n'
             '# With the GPU, MAnalyse searches with meander=QTGMC_gpu_meander unless meander is given.\n'
             + wrapper('MSuper', ['c'], sup_p)
             + wrapper('MAnalyse', ['c'], ana_p, {'meander': 'Default(meander, QTGMC_gpu_meander)'})
             + wrapper('MRecalculate', ['c', 'vectors'], rec_p))
    text += ('\n\n# GPU version of the search clip pre-filter (QTGMC "Motion Analysis" from bobbed to srchClip), equal to the CPU lines it\n'
             '# replaces; where those leave chroma undefined (masktools U=V=1, RemoveGrain -1 without ChromaMotion) the GPU copies it\n'
             '# (the luma-only analysis does not read it). download=false: token frames (only the Cuda* filters may consume them)\n'
             'function QTGMC_gpu_srchclip( clip c, int TR0, int Rep0, int SrchClipPP, bool ChromaMotion, bool RepChroma, int w, int h, \\\n'
             '                             float epsilon, bool download )\n'
             '{\n'
             '\tCMts  = ChromaMotion ? 255 : 0\n'
             '\tCMrg  = ChromaMotion ? 12  : -1\n'
             '\tCMgpu = ChromaMotion ? 3   : 2\n'
             '\tb   = c.CudaBob( 0, 0.5 )\n'
             '\tb   = GetParity(c) ? b.AssumeTFF() : b.AssumeBFF()  # sh_Bob\n'
             '\tts1 = TR0 == 1 ? b.CudaTemporalSoften( 1, 255, CMts, 28, 2 ) : NOP()\n'
             '\t# TR0 2: TemporalSoften 1 and 2 and the two merges in one step\n'
             '\tbin = TR0 == 0 ? b : TR0 == 1 ? (ChromaMotion ? ts1.CudaMerge( b, 0.25 ) : ts1.CudaMergeLuma( b, 0.25 )) : \\\n'
             '\t                                b.CudaQtgmcBinomial0( 255, CMts, 28, chroma=ChromaMotion )\n'
             '\tr0  = Rep0 == 0 ? bin : CudaKeepOnlyBobShimmerFixes( bin, b, Rep0, RepChroma && ChromaMotion )\n'
             '\tsb  = SrchClipPP == 1 ? r0.CudaBilinearResize( w/2, h/2 ).CudaRemoveGrain( 12, CMrg ).CudaBilinearResize( w, h ) : \\\n'
             '\t      SrchClipPP >= 2 ? r0.CudaRemoveGrain( 12, CMrg ).CudaGaussResize( w,h, 0,0, w+epsilon,h+epsilon, p=2 ) : NOP()\n'
             '\tsb  = SrchClipPP >= 2 ? (ChromaMotion ? sb.CudaMerge( r0, 0.1 ) : sb.CudaMergeLuma( r0, 0.1 )) : sb\n'
             '\ttw  = SrchClipPP == 3 ? CudaLutxy( r0, b, "x 3 scalef + y < x 3 scalef + x 3 scalef - y > x 3 scalef - y ? ?", use_expr=2, U=CMgpu,V=CMgpu ) : NOP()\n'
             '\ts   = SrchClipPP == 0 ? r0 : SrchClipPP < 3 ? sb : \\\n'
             '\t      CudaLutxy( sb, tw, "x 7 scalef + y < x 2 scalef + x 7 scalef - y > x 2 scalef - x 51 * y 49 * + 100 / ? ?", use_expr=2, U=CMgpu,V=CMgpu )\n'
             '\treturn download ? s.CudaDownload() : s\n'
             '}\n'
             '\n# NNEDI3 for QTGMC: the GPU port (its frames downloaded for the CPU filters that follow) or the plugin\n'
             'function QTGMC_gpu_nnedi3( clip c, int "field", int "nsize", int "nns", int "qual", int "pscrn", int "threads", bool "Y", bool "U", bool "V" )\n'
             '{\n'
             '\tg = QTGMC_gpu && QTGMC_gpu_edi ? c.CudaNNEDI3( field=field, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, threads=threads, Y=Y, U=U, V=V ) : NOP()\n'
             '\treturn !(QTGMC_gpu && QTGMC_gpu_edi) ? c.NNEDI3( field=field, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, threads=threads, Y=Y, U=U, V=V ) : \\\n'
             '\t       QTGMC_gpu_edi_token ? g : g.CudaDownload()\n'
             '}\n'
             '\n# The output side of QTGMC on the GPU, on the device frames of edi, from "Get the max/min value" to addNoise2 (lossed2 =\n'
             '# sharpLimit2 = repair2 for the structures QTGMC_gpuOut admits); equal to the CPU lines it replaces\n'
             'function QTGMC_gpu_outside( clip edi, clip bVec1, clip fVec1, val bVec2, val fVec2, val bVec3, val fVec3, int SubPel, int SubPelInterp, \\\n'
             '                            int hpad, int vpad, int ThSCD1, int ThSCD2, int ThSAD1, int ThSAD2, int TR1, int TR2, int SMode, bool Precise, \\\n'
             '                            int SLMode, int SLRad, int Sbb, string resharpExpr, int rgBlur, int Sovs, int Rep2, bool RepChroma, val finalNoise, \\\n'
             '                            string grainExpr, string noiseExpr, float GrainRestore, float NoiseRestore, int CNmt1, int CNmt2, \\\n'
             '                            float bbWidth, float bbHeight )\n'
             '{\n'
             '\tedi      = edi.CudaUpload()  # an interpolation made on the CPU goes to the device here; token frames pass through\n'
             '\tediSuper = edi.CudaMSuper( pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad )\n'
             '\ttsl      = SLMode == 2       # temporal limit right after the resharp (SLMode 4 is not admitted)\n'
             '\tbComp1   = tsl ? edi.CudaMCompensate( ediSuper, bVec1, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tfComp1   = tsl ? edi.CudaMCompensate( ediSuper, fVec1, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tbComp3   = tsl && SLRad > 1 ? edi.CudaMCompensate( ediSuper, bVec3, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tfComp3   = tsl && SLRad > 1 ? edi.CudaMCompensate( ediSuper, fVec3, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tnref     = !tsl ? 0 : SLRad > 1 ? 5 : 3\n'
             '\tdegrain1 = TR1 > 0 ? edi.CudaMDegrain1( ediSuper, bVec1, fVec1, thSAD=ThSAD1, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tdegrain2 = TR1 > 1 ? edi.CudaMDegrain1( ediSuper, bVec2, fVec2, thSAD=ThSAD1, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : edi\n'
             '\tsbb1     = Sbb == 1 || Sbb == 3\n'
             '\t# binomial1, vresharp, resharp and (without a back blend before it) the temporal limit in one step\n'
             '\tresharp  = CudaQtgmcSharpLimit( edi, degrain1, fComp1, bComp1, resharpExpr, rgBlur, Sovs, degrain2=degrain2, fComp3=fComp3, \\\n'
             '\t                                bComp3=bComp3, tr1=TR1, smode=SMode, precise=Precise, nref=sbb1 ? 0 : nref )\n'
             '\tlossed1  = Sbb > 0 ? CudaQtgmcSharpLimit( edi, degrain1, edi, edi, resharpExpr, rgBlur, 0, degrain2=degrain2, tr1=TR1, smode=0, nref=0 ) : NOP()\n'
             '\t# back blend before the limit (Sbb 1, 3), with the limit fused; back blend after it (Sbb 2, 3)\n'
             '\tsharpLimit1 = sbb1 ? CudaQtgmcBackBlend( resharp, lossed1, bbWidth, bbHeight, edi=edi, fComp1=fComp1, bComp1=bComp1, fComp3=fComp3, \\\n'
             '\t                                         bComp3=bComp3, nref=nref, sovs=Sovs ) : resharp\n'
             '\tbackBlend2  = Sbb >= 2 ? CudaQtgmcBackBlend( sharpLimit1, lossed1, bbWidth, bbHeight ) : sharpLimit1\n'
             '\taddNoise1   = GrainRestore > 0.0 ? CudaQtgmcAddNoise( backBlend2, finalNoise, grainExpr, use_expr=2, lutUV=CNmt1, addUV=CNmt2 ) : backBlend2\n'
             '\tstableSuper = TR2 > 0 ? addNoise1.CudaMSuper( pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad ) : NOP()\n'
             '\tstable      = TR2 == 0 ? addNoise1 : \\\n'
             '\t              TR2 == 1 ? addNoise1.CudaMDegrain1( stableSuper, bVec1, fVec1, thSAD=ThSAD2, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : \\\n'
             '\t              TR2 == 2 ? addNoise1.CudaMDegrain2( stableSuper, bVec1, fVec1, bVec2, fVec2, thSAD=ThSAD2, thSCD1=ThSCD1, thSCD2=ThSCD2 ) : \\\n'
             '\t                         addNoise1.CudaMDegrain3( stableSuper, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, thSAD=ThSAD2, thSCD1=ThSCD1, thSCD2=ThSCD2 )\n'
             '\trepair2     = (Rep2 == 0) ? stable : CudaKeepOnlyBobShimmerFixes( stable, edi, Rep2, RepChroma )\n'
             '\treturn NoiseRestore > 0.0 ? CudaQtgmcAddNoise( repair2, finalNoise, noiseExpr, use_expr=3, clamp_float=true, lutUV=CNmt1, addUV=CNmt2 ) : repair2\n'
             '}\n')
    if crlf:
        text = text.replace('\n', '\r\n')
    open(dst, 'wb').write(text.encode('latin-1'))
    print('wrote', dst)
    for k, v in n.items():
        print('  %-12s %d replacements' % (k, v))
    return 0


if __name__ == '__main__':
    sys.exit(main())
