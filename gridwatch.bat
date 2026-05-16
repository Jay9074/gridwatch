@echo off
set "CONDA=C:\anaconda3"
set "ENV=C:\anaconda3\envs\gridwatch"
set "PATH=%ENV%;%ENV%\Scripts;%ENV%\Library\bin;%CONDA%;%CONDA%\Scripts;%PATH%"
cd /d C:\projects\gridwatch-main
cmd /k