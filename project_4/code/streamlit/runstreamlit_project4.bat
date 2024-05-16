@echo off
cd c:/Users/Ryan/AppData/Roaming/Python/Python312/Scripts

if "%1" neq "" (
   streamlit.exe run %1
) else (
   streamlit.exe run "C:\Users\Ryan\GA\Github_Project4_Team6\DSI42-Project4-Team6\code\streamlit\LifestyleBuddy.py"
)


