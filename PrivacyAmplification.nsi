!define /date BuildTime "%Y-%m-%d"
Name "PrivacyAmplification"
Caption "PrivacyAmplification v1.1.0 by Nico Bosshard from ${BuildTime}"
Icon "Icon.ico"
OutFile "PrivacyAmplificationSetup.exe"
SetCompress auto
SetCompressor LZMA

SetDateSave on
SetDatablockOptimize on
CRCCheck on
SilentInstall normal
BGGradient 000000 800000 FFFFFF
InstallColors 80FF80 000030
XPStyle on

InstallDir "$PROGRAMFILES64\PrivacyAmplification"
InstallDirRegKey HKLM "Software\PrivacyAmplification" "Install_Dir"

CheckBitmap "${NSISDIR}\Contrib\Graphics\Checks\classic-cross.bmp"

LicenseText "PrivacyAmplification is a free open source software from Nico Bosshard"
LicenseData "LICENSE"

RequestExecutionLevel admin

Page license
Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

InstType "Full"
InstType "None"

AutoCloseWindow false
ShowInstDetails show

Section "" UninstallPrevious
  ExecWait '"$INSTDIR\Uninstall.exe" /S _?=$INSTDIR'
SectionEnd

Section ""
  SetOutPath "$INSTDIR"
  File Icon.ico
  WriteRegStr HKLM SOFTWARE\PrivacyAmplification "Install_Dir" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "DisplayName" "PrivacyAmplification"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteUninstaller "Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\PrivacyAmplification"
SectionEnd

SectionGroup /e "Components"
Section "PrivacyAmplification"
  SectionIn 1
  CreateDirectory "$INSTDIR\PrivacyAmplification"
  SetOutPath "$INSTDIR\PrivacyAmplification"
  File "PrivacyAmplification\bin\Release\config.yaml"
  File "PrivacyAmplification\bin\Release\PrivacyAmplification.exe"
  File "PrivacyAmplification\bin\Release\cufft64_10.dll"
  File "PrivacyAmplification\bin\Release\libzmq-v142-mt-4_3_3.dll"
  SetCompress off
  File "PrivacyAmplification\bin\Release\toeplitz_seed.bin"
  File "PrivacyAmplification\bin\Release\keyfile.bin"
  SetCompress auto
  CreateShortcut "$DESKTOP\PrivacyAmplification.lnk" "$INSTDIR\PrivacyAmplification\PrivacyAmplification.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\PrivacyAmplification.lnk" "$INSTDIR\PrivacyAmplification\PrivacyAmplification.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "MatrixSeedServerExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\MatrixSeedServerExample"
  SetOutPath "$INSTDIR\MatrixSeedServerExample"
  File "examples\MatrixSeedServerExample\x64\Release\MatrixSeedServerExample.exe"
  File "examples\MatrixSeedServerExample\x64\Release\libzmq-v142-mt-4_3_3.dll"
  SetCompress off
  File "examples\MatrixSeedServerExample\x64\Release\toeplitz_seed.bin"
  SetCompress auto
  CreateShortcut "$DESKTOP\MatrixSeedServerExample.lnk" "$INSTDIR\MatrixSeedServerExample\MatrixSeedServerExample.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\MatrixSeedServerExample.lnk" "$INSTDIR\MatrixSeedServerExample\MatrixSeedServerExample.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "SendKeysExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\SendKeysExample"
  SetOutPath "$INSTDIR\SendKeysExample"
  File "examples\SendKeysExample\x64\Release\SendKeysExample.exe"
  File "examples\SendKeysExample\x64\Release\libzmq-v142-mt-4_3_3.dll"
  SetCompress off
  File "examples\SendKeysExample\x64\Release\keyfile.bin"
  SetCompress auto
  CreateShortcut "$DESKTOP\SendKeysExample.lnk" "$INSTDIR\SendKeysExample\SendKeysExample.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\SendKeysExample.lnk" "$INSTDIR\SendKeysExample\SendKeysExample.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "ReceiveAmpOutExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\ReceiveAmpOutExample"
  SetOutPath "$INSTDIR\ReceiveAmpOutExample"
  File "examples\ReceiveAmpOutExample\x64\Release\ReceiveAmpOutExample.exe"
  File "examples\ReceiveAmpOutExample\x64\Release\libzmq-v142-mt-4_3_3.dll"
  CreateShortcut "$DESKTOP\ReceiveAmpOutExample.lnk" "$INSTDIR\ReceiveAmpOutExample\ReceiveAmpOutExample.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\ReceiveAmpOutExample.lnk" "$INSTDIR\ReceiveAmpOutExample\ReceiveAmpOutExample.exe" "" "$INSTDIR\Icon.ico"
SectionEnd
SectionGroupEnd

Section ""
  SetOutPath "$INSTDIR"
  WriteUninstaller "Uninstall.exe"
  WriteRegStr HKLM "SOFTWARE\PrivacyAmplification" "Install_Dir" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "DisplayName" "PrivacyAmplification"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "DisplayIcon" '"$INSTDIR\Icon.ico"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "Publisher" "Nico Bosshard"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "NoModify" 1
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification" "NoRepair" 1
SectionEnd


;--------------------------------
; Uninstaller
;--------------------------------
UninstallText "This will uninstall PrivacyAmplification. Hit next to continue."
UninstallIcon "${NSISDIR}\Contrib\Graphics\Icons\nsis1-uninstall.ico"

Section "Uninstall"
  Delete "$DESKTOP\PrivacyAmplification.lnk"
  Delete "$DESKTOP\MatrixSeedServerExample.lnk"
  Delete "$DESKTOP\SendKeysExample.lnk"
  Delete "$DESKTOP\ReceiveAmpOutExample.lnk"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\PrivacyAmplification"
  DeleteRegKey HKLM "SOFTWARE\PrivacyAmplification"
  RMDir /r "$SMPROGRAMS\PrivacyAmplification"
  RMDir /r "$INSTDIR"
SectionEnd
