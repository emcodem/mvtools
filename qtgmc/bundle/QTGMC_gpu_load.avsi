# Loads the GPU QTGMC bundle from the folder this file is in: Import it, then call QTGMC as usual.
#
#   Import("<bundle folder>\QTGMC_gpu_load.avsi")
#   QTGMC_gpu_on()                         # GPU for everything that has a GPU version (default after this Import: off)
#   c.QTGMC(Preset="Fast")
#
# All DLLs come from .\plugins (mvtools2.dll, mvtools_cuda.dll, cudafilters.dll and avstp.dll must stay together and come
# from one build). FFT3DFilter (QTGMC Very Slow / Placebo noise processing) needs the FFTW library: it is loaded here with
# LoadDLL, so nothing has to be added to PATH.

QTGMC_gpu_bundle = ScriptDir()
QTGMC_gpu_plugins = QTGMC_gpu_bundle + "plugins\"
LoadPlugin(QTGMC_gpu_plugins + "LoadDLL.dll")
LoadDLL(QTGMC_gpu_plugins + "libfftw3f-3.dll")
LoadPlugin(QTGMC_gpu_plugins + "mvtools2.dll")
LoadPlugin(QTGMC_gpu_plugins + "cudafilters.dll")
LoadPlugin(QTGMC_gpu_plugins + "ffms2.dll")
LoadPlugin(QTGMC_gpu_plugins + "masktools2.dll")
LoadPlugin(QTGMC_gpu_plugins + "RgTools.dll")
LoadPlugin(QTGMC_gpu_plugins + "plugins_JPSDR.dll")
LoadPlugin(QTGMC_gpu_plugins + "fft3dfilter.dll")
LoadPlugin(QTGMC_gpu_plugins + "yadifmod2.dll")
Import(QTGMC_gpu_bundle + "scripts\mtmodes.avsi")
Import(QTGMC_gpu_bundle + "scripts\Zs_RF_Shared.avsi")
Import(QTGMC_gpu_bundle + "scripts\SMDegrain.avsi")
Import(QTGMC_gpu_bundle + "scripts\QTGMC_gpu.avsi")

# every GPU switch on: search clip, motion analysis, NNEDI3 and the output side on the GPU (meander=false motion search)
function QTGMC_gpu_on()
{
	global QTGMC_gpu = true
	global QTGMC_gpu_search = true
	global QTGMC_gpu_tokens = true
	global QTGMC_gpu_edi = true
	global QTGMC_gpu_out = true
	return true
}

# every GPU switch off: QTGMC is the stock CPU QTGMC
function QTGMC_gpu_off()
{
	global QTGMC_gpu = false
	global QTGMC_gpu_search = false
	global QTGMC_gpu_tokens = false
	global QTGMC_gpu_edi = false
	global QTGMC_gpu_out = false
	return true
}
