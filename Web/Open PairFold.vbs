' One-click PairFold — double-click to open the web app (no console).
' Location: Web\Open PairFold.vbs

Option Explicit
Dim sh, fso, webDir, script

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
webDir = fso.GetParentFolderName(WScript.ScriptFullName)
script = fso.BuildPath(webDir, "open_pairfold.py")
sh.CurrentDirectory = webDir

If Not fso.FileExists(script) Then
  MsgBox "Missing open_pairfold.py in Web\", vbCritical, "PairFold"
  WScript.Quit 1
End If

On Error Resume Next
sh.Run "pythonw """ & script & """", 0, False
If Err.Number <> 0 Then
  Err.Clear
  sh.Run "python """ & script & """", 0, False
  If Err.Number <> 0 Then
    MsgBox "Python was not found on PATH. Install Python 3 and retry.", vbCritical, "PairFold"
    WScript.Quit 1
  End If
End If
