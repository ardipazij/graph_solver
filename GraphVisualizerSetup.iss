[Setup]
AppName=Визуализатор графов
AppVersion=1.0
AppPublisher=ardipzaijInc
DefaultDirName={pf}\GraphVisualizer
DefaultGroupName=Визуализатор графов
OutputDir=.
OutputBaseFilename=GraphVisualizerSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Files]
Source: "C:\Users\arxie\OneDrive\Рабочий стол\graph_solver-develop\graph_solver-develop\src\dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Визуализатор графов"; Filename: "{app}\graph_solver.exe"
Name: "{group}\Удалить Визуализатор графов"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Визуализатор графов"; Filename: "{app}\graph_solver.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительные задачи:"

[Run]
Filename: "{app}\graph_solver.exe"; Description: "Запустить Визуализатор графов"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"