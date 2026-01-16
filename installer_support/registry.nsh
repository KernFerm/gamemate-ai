; NSIS helper to write registry keys and uninstall entries for GameMate AI
; This file is intended to be included by electron/build/installer.nsh

!macro customInstall
  DetailPrint "[installer_support] Writing registry entries for GameMate AI"

  ; Write install dir and version (HKLM when $ALLUSERS==1, otherwise HKCU)
  StrCmp $ALLUSERS "1" 0 +3
    WriteRegStr HKLM "Software\GameMate AI" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\GameMate AI" "Version" "1.0.0"
    Goto _reg_done
  WriteRegStr HKCU "Software\GameMate AI" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\GameMate AI" "Version" "1.0.0"
  _reg_done:

  ; Add uninstall information so Windows shows app in Programs & Features
  ; Use a safe key name without spaces for registry subkey
  StrCpy $0 "GameMate_AI_Assistant"
  ; Add uninstall information (HKLM when all-users, otherwise HKCU)
  StrCmp $ALLUSERS "1" 0 +8
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "DisplayName" "GameMate AI"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "DisplayVersion" "1.0.0"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "Publisher" "GameMate"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "UninstallString" '"$INSTDIR\\Uninstall.exe"'
    Goto _unreg_done
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "DisplayName" "GameMate AI"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "DisplayVersion" "1.0.0"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "Publisher" "GameMate"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0" "UninstallString" '"$INSTDIR\\Uninstall.exe"'
  _unreg_done:

  ; Optional: register file association or protocol here if needed
  ; Example: WriteRegStr HKCU "Software\Classes\msigaming" "" "URL:MSI Gaming Protocol"

!macroend

!macro customUnInstall
  DetailPrint "[installer_support] Removing registry entries for GameMate AI"

  ; Remove application-specific key (HKLM when all-users, otherwise HKCU)
  StrCmp $ALLUSERS "1" 0 +3
    DeleteRegKey HKLM "Software\GameMate AI"
    Goto _del_uninstall
  DeleteRegKey HKCU "Software\GameMate AI"
  _del_uninstall:

  ; Remove uninstall registration
  StrCpy $0 "GameMate_AI_Assistant"
  StrCmp $ALLUSERS "1" 0 +3
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0"
    Goto _del_done
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$0"
  _del_done:

  ; If you created file associations or protocols, remove them here
!macroend
