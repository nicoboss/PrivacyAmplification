az devops login --organization https://dev.azure.com/nicoboss/
az artifacts universal publish ^
    --organization https://dev.azure.com/nicoboss/ ^
    --project="PrivacyAmplification" ^
    --scope project ^
    --feed Libs ^
    --name cufft64_10.dll ^
    --version 1.1.0 ^
    --description "CUDA Fast Fourier Transform library" ^
    --path .


az artifacts universal publish ^
    --organization https://dev.azure.com/nicoboss/ ^
    --project="PrivacyAmplification" ^
    --scope project ^
    --feed Libs ^
    --name libzmq-v142-mt-4_3_3.dll ^
    --version 1.1.0 ^
    --description "ZeroMQ core engine in C++" ^
    --path .


az artifacts universal publish ^
    --organization https://dev.azure.com/nicoboss/ ^
    --project="PrivacyAmplification" ^
    --scope project ^
    --feed Testdata ^
    --name keyfile.bin ^
    --version 1.0.0 ^
    --description "keyfile.bin" ^
    --path .

az artifacts universal publish ^
    --organization https://dev.azure.com/nicoboss/ ^
    --project="PrivacyAmplification" ^
    --scope project ^
    --feed Testdata ^
    --name toeplitz_seed.bin ^
    --version 1.0.0 ^
    --description "toeplitz_seed.bin" ^
    --path .
