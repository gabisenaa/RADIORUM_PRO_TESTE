# RADIORUM Pro — Lightweight, fast DICOM 3D Viewer (Prototype)

Goals:
- Open any CT/MR DICOM series quickly (< 1 minute) and provide smooth interactive 3D viewing on mobile and desktop.
- Lightweight meshes via aggressive but controlled downsampling + vertex-clustering decimation.
- Adjustable threshold without reuploading the original DICOMs.

How it works (high level):
1. User uploads a ZIP of DICOM files or multiple files.
2. Server reads slices, stacks into volume and saves a compressed .npz case file.
3. When a viewer requests a mesh, server:
   - Loads volume, downsamples to target resolution based on chosen quality
   - Normalizes and runs marching_cubes (scikit-image)
   - Applies voxel-grid vertex clustering to reduce vertex count
   - Returns compact JSON (rounded floats, integer indices)

Deployment (Render):
- Push this repo to GitHub and create a Web Service (Python).
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

Production notes (must address before clinical use):
- Add authentication, TLS enforcement, audit logging, PHI handling and consent.
- Add automated medical QC and unit/regression tests with reference datasets.
- Consider using SimpleITK/GDCM for compressed DICOM support in production.
