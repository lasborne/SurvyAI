; =====================================================================
; SurvyAI Windows Installer (Inno Setup)
; =====================================================================
; Build the app first:
;     pyinstaller --noconfirm --clean survyai.spec
; Then compile this installer:
;     ISCC.exe installer\survyai.iss /DAppVersion=1.0.0
; Output:
;     installer\Output\SurvyAI-Setup-<version>.exe
;
; AppVersion may be passed on the command line (/DAppVersion=...). If omitted it
; defaults to the value below — keep it in sync with survyai/version.py.
; =====================================================================

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

#define AppName "SurvyAI"
#define AppPublisher "SurvyAI"
#define AppExeName "SurvyAI.exe"
; Stable AppId (GUID) so upgrades replace the same installed product.
#define AppId "{{B6F2A8E1-7C4D-4E59-9A3B-5D2F1C8E0A77}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-machine install requires elevation; switch to "lowest" for per-user.
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir=Output
OutputBaseFilename=SurvyAI-Setup-{#AppVersion}
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
; SetupIconFile=survyai.ico   ; add a real .ico to brand the installer

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle the entire PyInstaller one-dir output (dist\SurvyAI\*).
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove anything the app created inside its install dir (logs/caches if any).
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
