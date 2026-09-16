Set-Location "C:\Users\Ajay\Documents\Thesis"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python .\scripts\analyse_architecture_suite.py .\runs\thesis_architecture_suite_v1
