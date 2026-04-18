@echo off
echo Installing python-pptx if needed...
cd /d "c:\Users\gunav\Downloads\Mtech_2025_Admission\IITK\MTech\Sem2\AML\Project"
venv\Scripts\pip install python-pptx --quiet
echo.
echo Generating PPT...
venv\Scripts\python.exe scratch\generate_ppt.py
echo.
pause
