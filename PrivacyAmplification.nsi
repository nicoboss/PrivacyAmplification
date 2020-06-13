Name "PrivacyAmplification"
Caption "PrivacyAmplification - Nico Bosshard"
Icon "Icon.ico"
OutFile "PrivacyAmplificationSetup.exe"
;SetCompress off
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
  File /r /x *.exp /x *.lib /x *.pdb /x *.iobj /x *.ipdb "PrivacyAmplification\bin\Release\"
  CreateShortcut "$DESKTOP\PrivacyAmplification.lnk" "$INSTDIR\PrivacyAmplification\PrivacyAmplification.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\PrivacyAmplification.lnk" "$INSTDIR\PrivacyAmplification\PrivacyAmplification.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "MatrixSeedServerExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\MatrixSeedServerExample"
  SetOutPath "$INSTDIR\MatrixSeedServerExample"
  File /r /x *.exp /x *.lib /x *.pdb /x *.iobj /x *.ipdb "MatrixSeedServerExample\x64\Release\"
  CreateShortcut "$DESKTOP\MatrixSeedServerExample.lnk" "$INSTDIR\MatrixSeedServerExample\MatrixSeedServerExample.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\MatrixSeedServerExample.lnk" "$INSTDIR\MatrixSeedServerExample\MatrixSeedServerExample.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "SendKeysExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\SendKeysExample"
  SetOutPath "$INSTDIR\SendKeysExample"
  File /r /x *.exp /x *.lib /x *.pdb /x *.iobj /x *.ipdb "SendKeysExample\x64\Release\"
  CreateShortcut "$DESKTOP\SendKeysExample.lnk" "$INSTDIR\SendKeysExample\SendKeysExample.exe" "" "$INSTDIR\Icon.ico"
  CreateShortcut "$SMPROGRAMS\PrivacyAmplification\SendKeysExample.lnk" "$INSTDIR\SendKeysExample\SendKeysExample.exe" "" "$INSTDIR\Icon.ico"
SectionEnd

Section "ReceiveAmpOutExample"
  SectionIn 1
  CreateDirectory "$INSTDIR\ReceiveAmpOutExample"
  SetOutPath "$INSTDIR\ReceiveAmpOutExample"
  File /r /x *.exp /x *.lib /x *.pdb /x *.iobj /x *.ipdb "ReceiveAmpOutExample\x64\Release\"
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
