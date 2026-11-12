# Installation check: one line per engine and RESULT: PASS / FAIL
LoadPlugin("CudaConvolution.dll")
report = CudaConvolution_SelfTest() + Chr(10) + CudaConvolution_Version()
BlankClip(width=1600, height=200, length=1, pixel_type="YV24")
Subtitle(ReplaceStr(report, Chr(10), "\n"), lsp=0, size=22)     # Subtitle breaks lines at "\n"
