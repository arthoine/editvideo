@echo off
echo ========================================
echo   Nettoyage complet d'EditVideo
echo ========================================
echo.

REM Nettoyer le cache Python
if exist src\__pycache__ (
    echo [1/4] Suppression du cache Python src...
    rmdir /s /q src\__pycache__
)
if exist __pycache__ (
    echo [2/4] Suppression du cache Python root...
    rmdir /s /q __pycache__
)

REM Nettoyer le cache d'analyse
if exist .cache (
    echo [3/4] Suppression du cache d'analyse (.cache)...
    rmdir /s /q .cache
)

REM Nettoyer les fichiers temporaires
if exist "%TEMP%\editvideo" (
    echo [4/4] Suppression des fichiers temporaires...
    rmdir /s /q "%TEMP%\editvideo" 2>nul
)

echo.
echo ========================================
echo   Nettoyage termine !
echo ========================================
echo.
echo Vous pouvez maintenant relancer le script en mode clean.
echo.
pause
