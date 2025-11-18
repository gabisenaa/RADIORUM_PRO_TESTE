import os, tempfile, zipfile, json, uuid, math
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import numpy as np
import pydicom
from skimage import measure
from pathlib import Path

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
DATA = BASE / "data"
MESH = BASE / "mesh"
for d in (UPLOADS, DATA, MESH):
    d.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------- Utilities ----------------------
def read_dicom_series(file_paths):
    slices = []
    for p in file_paths:
        try:
            ds = pydicom.dcmread(p)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception as e:
            # skip files that can't be read
            print("skip", p, e)
    if not slices:
        raise RuntimeError("No valid DICOM slices found")
    # sort
    def keyfn(s):
        if hasattr(s, "InstanceNumber"):
            return int(s.InstanceNumber)
        if hasattr(s, "SliceLocation"):
            return float(s.SliceLocation)
        return 0
    slices.sort(key=keyfn)
    arrs = [s.pixel_array.astype(np.int16) for s in slices]
    volume = np.stack(arrs, axis=0)
    # spacing fallback
    try:
        spacing = (float(slices[0].SliceThickness), float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]))
    except Exception:
        spacing = (1.0,1.0,1.0)
    return volume, spacing

def resample_to_target(volume, spacing, target_max_dim=128):
    # keep proportions, find scaling factor so largest axis becomes target_max_dim
    z,y,x = volume.shape
    max_axis = max(z,y,x)
    if max_axis <= target_max_dim:
        return volume, spacing
    scale = target_max_dim / float(max_axis)
    new_shape = (max(1, int(round(z*scale))), max(1, int(round(y*scale))), max(1, int(round(x*scale))))
    # use simple downsampling by block-mean for speed
    vol = volume
    factors = (z//new_shape[0], y//new_shape[1], x//new_shape[2])
    # if factors are 1 or less, fallback to numpy resize by slicing
    if any(f < 1 for f in factors):
        # use numpy zoom (approx) via slicing/resizing
        from skimage.transform import resize
        vol_resized = resize(vol.astype(float), new_shape, order=1, preserve_range=True, anti_aliasing=True)
        return vol_resized.astype(vol.dtype), spacing
    # block mean downsample
    vol_cropped = vol[:factors[0]*new_shape[0], :factors[1]*new_shape[1], :factors[2]*new_shape[2]]
    vol_reshaped = vol_cropped.reshape(new_shape[0], factors[0], new_shape[1], factors[1], new_shape[2], factors[2])
    vol_down = vol_reshaped.mean(axis=(1,3,5))
    # adjust spacing inversely
    spacing_new = (spacing[0]* (vol.shape[0]/vol_down.shape[0]), spacing[1]* (vol.shape[1]/vol_down.shape[1]), spacing[2]* (vol.shape[2]/vol_down.shape[2]))
    return vol_down.astype(vol.dtype), spacing_new

def generate_mesh(volume, spacing, threshold=0.5):
    # normalize to 0..1
    v = volume.astype(float)
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    # marching cubes; may raise on flat volumes
    verts, faces, normals, vals = measure.marching_cubes(v, level=threshold, spacing=spacing)
    return verts, faces

def vertex_clustering(verts, faces, cluster_size=1.0):
    # cluster vertices into voxels of size cluster_size in value units
    # returns new_verts, new_faces
    if len(verts)==0:
        return verts, faces
    # quantize coords
    q = (verts / cluster_size).round().astype(np.int64)
    # map unique quantized coords to averaged vertex
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    # compute averaged positions per cluster
    new_verts = np.zeros((uniq.shape[0], 3), dtype=np.float32)
    for i in range(uniq.shape[0]):
        new_verts[i] = verts[inv==i].mean(axis=0)
    # rebuild faces using inv mapping; remove degenerate faces
    new_faces = []
    for f in faces:
        a,b,c = inv[f[0]], inv[f[1]], inv[f[2]]
        if a==b or b==c or a==c:
            continue
        new_faces.append([int(a),int(b),int(c)])
    if not new_faces:
        return new_verts, np.zeros((0,3), dtype=np.int32)
    return new_verts, np.array(new_faces, dtype=np.int32)

def mesh_to_compact_json(verts, faces, max_precision=3):
    # round floats to reduce payload and convert to lists
    verts_round = np.round(verts.astype(float), decimals=max_precision).tolist()
    faces_list = faces.astype(int).tolist()
    return {"vertices": verts_round, "faces": faces_list}

# ---------------------- Routes ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # accept zip or multiple files
    if "zipfile" in request.files and request.files["zipfile"].filename!="":
        z = request.files["zipfile"]
        tmp = tempfile.mkdtemp(dir=str(UPLOADS))
        zpath = os.path.join(tmp, "upload.zip")
        z.save(zpath)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmp)
        files = []
        for root,_,fnames in os.walk(tmp):
            for fn in fnames:
                files.append(os.path.join(root, fn))
    else:
        fls = request.files.getlist("files")
        if not fls:
            return "No files", 400
        tmp = tempfile.mkdtemp(dir=str(UPLOADS))
        files = []
        for f in fls:
            out = os.path.join(tmp, f.filename)
            f.save(out)
            files.append(out)
    try:
        volume, spacing = read_dicom_series(files)
    except Exception as e:
        return f"Failed to read DICOM: {e}", 400
    case_id = str(uuid.uuid4())
    # save compressed case
    np.savez_compressed(DATA / f"{case_id}.npz", volume=volume, spacing=spacing)
    # generate a quick low-res mesh for immediate viewing
    vol_low, sp_low = resample_to_target(volume, spacing, target_max_dim=96)
    try:
        verts, faces = generate_mesh(vol_low, sp_low, threshold=0.5)
    except Exception as e:
        # fallback: try tiny target
        vol_tiny, sp_tiny = resample_to_target(volume, spacing, target_max_dim=64)
        verts, faces = generate_mesh(vol_tiny, sp_tiny, threshold=0.5)
    # apply clustering decimation tuned for quick response
    verts_c, faces_c = vertex_clustering(verts, faces, cluster_size=max( max(sp_low), 1.0 ) * 1.5 )
    mesh_obj = mesh_to_compact_json(verts_c, faces_c, max_precision=3)
    with open(MESH / f"{case_id}.json", "w") as fh:
        json.dump(mesh_obj, fh)
    return redirect(url_for("viewer", case_id=case_id))

@app.route("/viewer/<case_id>")
def viewer(case_id):
    return render_template("viewer.html", case_id=case_id)

@app.route("/mesh/<case_id>.json")
def mesh_endpoint(case_id):
    # quality: low|med|high ; threshold 0..1
    q = request.args.get("quality","med")
    try:
        threshold = float(request.args.get("threshold","0.5"))
    except:
        threshold = 0.5
    case_file = DATA / f"{case_id}.npz"
    if not case_file.exists():
        return jsonify({"error":"case not found"}), 404
    # choose target resolution and clustering per quality
    if q=="low":
        target = 96; cluster=2.0; precision=2
    elif q=="high":
        target = 192; cluster=0.5; precision=4
    else:
        target = 128; cluster=1.0; precision=3
    # load volume
    with np.load(case_file, allow_pickle=True) as d:
        vol = d["volume"]; spacing = tuple(d["spacing"].tolist()) if hasattr(d["spacing"], "tolist") else tuple(d["spacing"])
    # resample quickly
    vol_r, sp_r = resample_to_target(vol, spacing, target_max_dim=target)
    try:
        verts, faces = generate_mesh(vol_r, sp_r, threshold=threshold)
    except Exception as e:
        return jsonify({"error":"marching cubes failed","msg":str(e)}), 500
    # decimate
    verts_d, faces_d = vertex_clustering(verts, faces, cluster_size=max(cluster, max(sp_r)))
    mesh = mesh_to_compact_json(verts_d, faces_d, max_precision=precision)
    return jsonify(mesh)

@app.route("/cases")
def list_cases():
    cs = [p.stem for p in DATA.glob("*.npz")]
    return jsonify({"cases": cs})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
