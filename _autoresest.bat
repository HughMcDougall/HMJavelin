@echo off
set /p "uinp=Pull current github repo? [Yes/*]: "
if %uinp%==Yes (git reset --hard origin/main) 
