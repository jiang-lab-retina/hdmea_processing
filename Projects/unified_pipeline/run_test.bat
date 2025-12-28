@echo off
REM Activate conda environment and run the full pipeline test
cd /d M:\Python_Project\Data_Processing_2027
call C:\ProgramData\anaconda3\Scripts\activate.bat hdmea

echo ======================================================================
echo Running Step 1-7 (from CMCR/CMTR)...
echo ======================================================================
python Projects/unified_pipeline/run_steps_1_to_7.py --dataset 2024.08.08-10.40.20-Rec

echo.
echo ======================================================================
echo Running Steps 8-end (from HDF5)...
echo ======================================================================
python Projects/unified_pipeline/run_steps_8_to_end.py --input Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec_steps1-7.h5

echo.
echo ======================================================================
echo Running comparison...
echo ======================================================================
python Projects/unified_pipeline/compare_outputs.py

echo.
echo Done! Press any key to exit.
pause

