@echo off
echo Checking for AzureCLI and azure-devops...
WHERE az > nul
IF %ERRORLEVEL% NEQ 0 (
  powershell -command "Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi"
  powershell -command "Get-Location; Start-Process msiexec.exe -verb runAs -Wait -ArgumentList '/I "%cd%"\AzureCLI.msi /quiet'"
  del /f .\AzureCLI.msi
  call az extension add --name azure-devops
)
set PATH=%PATH%;"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\"

echo Login to Azure DevOps...
echo 73fvfjbljqa7h7l5asob5rxbggay5lgva6osjinomk7hwlbosifa | az devops login --organization https://dev.azure.com/nicoboss/

echo Downloading assets from Azure DevOps...
call az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Libs" ^
  --name "cufft64_10.dll" ^
  --version "2.0.0" ^
  --path .

call az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Libs" ^
  --name "libzmq-v142-mt-4_3_5.dll" ^
  --version "2.0.0" ^
  --path .

call az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Testdata" ^
  --name "keyfile.bin" ^
  --version "2.0.0" ^
  --path .

call az artifacts universal download ^
  --organization "https://dev.azure.com/nicoboss/" ^
  --project "d47efac3-15b4-4aae-9e4d-6a4c0a44a420" ^
  --scope project ^
  --feed "Testdata" ^
  --name "toeplitz_seed.bin" ^
  --version "2.0.0" ^
  --path .
