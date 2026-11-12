# Example plugin set for UConvert (FFAStrans 1.4.2 bundle + tools + MxfSource + GPU plugins). Copy it to
# load_plugins.avs (not tracked: one machine's paths), which the tests and build_dist.py read, and set the paths.
# FFAStrans jobs load the same plugins from their own bundle; adjust the paths there.
# GPU set (G): mvtools2 / mvtools_cuda / cudafilters / splatfps from the mvtools fork, branch gpu_work, commit df02cdd
# (API 31), as in the A4000 bench package; they must come from one build and sit in one folder. mvtools2 loads
# mvtools_cuda.dll and avstp.dll from its own folder at run time (build_dist.py copies them along).
# QTGMC_gpu.avsi = QTGMC 3.393s patched by make_qtgmc_gpu.py; with its GPU switches off it is the stock CPU QTGMC.
FF = "C:/path/to/ffastrans/Processors/avs_plugins"
T  = "C:/path/to/tools"   # folder with lsmash/x64/LSMASHSource.dll (+ lsmash/AviSynth/LICENSE)
G  = "C:/path/to/gpu_package"   # plugins/ (mvtools2, mvtools_cuda, cudafilters, splatfps), scripts/ (QTGMC_gpu, Zs_RF_Shared)
CC = "C:/path/to/CudaConvolution"
LoadPlugin("C:/path/to/mxfsource.dll")
LoadPlugin(T + "/lsmash/x64/LSMASHSource.dll")
LoadPlugin(G + "/plugins/mvtools2.dll")
LoadPlugin(G + "/plugins/cudafilters.dll")
LoadPlugin(G + "/plugins/splatfps.dll")
LoadPlugin(CC + "/CudaConvolution.dll")
LoadPlugin(FF + "/MaskTools/x64/masktools2.dll")
LoadPlugin(FF + "/RgTools/x64/RgTools.dll")
LoadPlugin(FF + "/JPSDR/x64/plugins_JPSDR.dll")
LoadPlugin(FF + "/NNEDI3CL/NNEDI3CL.dll")
LoadPlugin(FF + "/bwdif/x64/BWDIF.dll")
LoadPlugin(FF + "/yadif17/x64/yadifmod2.dll")
LoadPlugin(FF + "/avsresize/x64/avsresize.dll")
Import(G + "/scripts/Zs_RF_Shared.avsi")
Import(G + "/scripts/QTGMC_gpu.avsi")
