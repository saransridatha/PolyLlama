# Build llama.cpp server on Windows with CMake.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Src = if ($env:LLAMA_CPP_SRC) { $env:LLAMA_CPP_SRC } else { Join-Path $Root "vendor\llama.cpp" }
$Build = if ($env:LLAMA_BUILD_DIR) { $env:LLAMA_BUILD_DIR } else { Join-Path $Root "vendor\llama-build" }

if (-not (Test-Path (Join-Path $Src "CMakeLists.txt"))) {
    Write-Host "Cloning llama.cpp…"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git $Src
}

$Cuda = @("-DGGML_CUDA=OFF")
if ($env:MS_LLAMA_CUDA -eq "1" -or ($env:MS_LLAMA_CUDA -ne "0" -and (Get-Command nvcc -ErrorAction SilentlyContinue))) {
    Write-Host "Enabling CUDA backend"
    $Cuda = @("-DGGML_CUDA=ON")
}

cmake -S $Src -B $Build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON @Cuda
cmake --build $Build --config Release --target llama-server -- /m

$Server = Join-Path $Build "bin\Release\llama-server.exe"
if (Test-Path $Server) {
    Write-Host "llama-server ready: $Server"
    & $Server --version
} else {
    throw "Build completed but $Server was not found."
}
