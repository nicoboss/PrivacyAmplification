WHERE az
IF %ERRORLEVEL% NEQ 0 powershell -command "Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi; Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'; rm .\AzureCLI.msi"
echo 73fvfjbljqa7h7l5asob5rxbggay5lgva6osjinomk7hwlbosifa | az devops login --organization https://dev.azure.com/nicoboss/
az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Libs" ^
  --name "cufft64_10.dll" ^
  --version "1.1.0" ^
  --path .

az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Libs" ^
  --name "libzmq-v142-mt-4_3_3.dll" ^
  --version "1.1.0" ^
  --path .

az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Testdata" ^
  --name "keyfile.bin" ^
  --version "1.0.0" ^
  --path .

az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Testdata" ^
  --name "toeplitz_seed.bin" ^
  --version "1.0.0" ^
  --path .


dir
