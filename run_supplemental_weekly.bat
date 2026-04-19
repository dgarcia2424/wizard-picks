@echo off
cd /d "C:\Users\garci\OneDrive\Documents\Claude\mlb_model_pipeline_dan"
C:\Users\garci\AppData\Local\Programs\Python\Python314\python.exe -X utf8 supplemental_pull.py --force-year 2026
C:\Users\garci\AppData\Local\Programs\Python\Python314\python.exe -X utf8 build_backtest.py --year 2026
