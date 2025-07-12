**Prompt for Windsurf – Initialize LiDAR PlotSafe Project Structure & Virtual Environment**

**Objective:**  
Set up a fresh, fully-configured LiDAR_PlotSafe project workspace on Windows 11, including Git initialization, skeleton directories, blank configuration files, a Python virtual environment, and installation of core dependencies—while preserving existing `.windsurf/` and `workflows/` folders.

---

**Instructions:**

1. **Environment**  
   - Operating System: Windows 11  
   - Shell: Command Prompt (`cmd.exe`)  

2. **Repository & Skeleton**  
   - Navigate to the root of the current VS Code workspace (`LiDAR PS`).  
   - Create a new folder `LiDAR_PlotSafe` alongside `.windsurf/` and `workflows/`.  
   - Inside `LiDAR_PlotSafe`, run:
     ```bat
     git init
     mkdir src\pipeline tests docs
     ```
   - Create empty files in `LiDAR_PlotSafe\`:
     ```bat
     type NUL > README.md
     type NUL > config.yaml
     type NUL > requirements.txt
     type NUL > LICENSE
     type NUL > .gitignore
     ```

3. **Populate `.gitignore`**  
   - Open `LiDAR_PlotSafe\.gitignore` and add:
     ```
     __pycache__/
     *.py[cod]
     .venv/
     venv/
     dist/
     build/
     *.egg-info/
     ```
   - Save the file.

4. **First Commit**  
   - From `LiDAR_PlotSafe\`, stage and commit:
     ```bat
     git add .
     git commit -m "Project skeleton: directories and empty files"
     ```

5. **Virtual Environment**  
   - Still in `LiDAR_PlotSafe\`, run:
     ```bat
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - Verify interpreter path:
     ```bat
     where python
     ```
     – the first path must be `…\LiDAR_PlotSafe\.venv\Scripts\python.exe`.

6. **Install Core Dependencies**  
   - Edit `requirements.txt` in `LiDAR_PlotSafe\` to include:
     ```
     laspy
     open3d
     pyntcloud
     csf-python
     numpy
     scipy
     pandas
     scikit-learn
     matplotlib
     torch
     tensorflow
     ```
   - Save and run:
     ```bat
     pip install -r requirements.txt
     ```

7. **Optional Extras**  
   - If needed later, install:
     ```bat
     pip install pdal python-pcl
     ```

8. **Final Commit**  
   - Stage the venv metadata and updated files:
     ```bat
     git add requirements.txt .gitignore
     git commit -m "Add virtual environment and core dependencies"
     ```

9. **Verification**  
   - Run a quick import test:
     ```bat
     python -c "import laspy, open3d, numpy; print('OK')"
     ```
   - Ensure no errors and that your prompt remains inside `LiDAR_PlotSafe/`.

---

**Expected Result:** ✅  
A new folder `LiDAR_PlotSafe/` coexists with `.windsurf/` and `workflows/`, contains the prescribed directory layout and files, has an active `.venv` with all core libraries installed, and two clean Git commits reflecting the skeleton and dependency setup.
