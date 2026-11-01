Set shell = CreateObject("WScript.Shell")
project = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
command = "powershell.exe -NoProfile -WindowStyle Hidden -Command ""Set-Location -LiteralPath '" & project & "'; uv run streamlit run src/chatbot_ui/app.py"""
shell.Run command, 0, False
