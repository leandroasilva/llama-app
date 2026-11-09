; Inno Setup script for the portable Windows build produced by package.sh.
; build-installer.ps1 sets LLAMA_VERSION, LLAMA_SOURCE_DIR and LLAMA_OUTPUT_DIR.

#ifndef LLAMA_VERSION
  #define LLAMA_VERSION GetEnv("LLAMA_VERSION")
#endif
#ifndef LLAMA_SOURCE_DIR
  #define LLAMA_SOURCE_DIR GetEnv("LLAMA_SOURCE_DIR")
#endif
#ifndef LLAMA_OUTPUT_DIR
  #define LLAMA_OUTPUT_DIR GetEnv("LLAMA_OUTPUT_DIR")
#endif

#define AppName "Llama"
#define AppExeName "llama-gtk.exe"
#define AppURL "https://github.com/leandroasilva/llama-app"

[Setup]
AppId={{7E3B1C2A-5D4F-4A6B-9C8D-1E2F3A4B5C6D}
AppName={#AppName}
AppVersion={#LLAMA_VERSION}
AppVerName={#AppName} {#LLAMA_VERSION}
AppPublisher=Llama
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir={#LLAMA_OUTPUT_DIR}
OutputBaseFilename=llama-gtk-{#LLAMA_VERSION}-windows-x86_64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#LLAMA_SOURCE_DIR}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
