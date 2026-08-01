# Archived experiments

These folders are retained only for provenance and comparison:

- `legacy-blender/` contains the former server-side render prototype.
- `legacy-colab/` contains the former notebook runner.
- `legacy-react-ui/` contains the abandoned duplicate React/Vite surface.

They are not imported, served, or required by the active application. The supported
runtime is the root `app.py` entrypoint, which serves `frontend/` and imports the
browser-first backend from `backend/`.
