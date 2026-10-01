import os
import shutil
from .runtime import frames, verify_frames
from .upscaler import compatibility, parallel_frames
from .config import cpu_count

_detector = None
_restorer = None
_weights = None


def init_faces(weights, threads):
    global _detector, _weights
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    compatibility()
    _weights = weights
    os.chdir(os.path.dirname(weights))
    _detector = cv2.dnn.readNetFromCaffe(weights + "/deploy.prototxt",
                                       weights + "/res10_300x300_ssd_iter_140000.caffemodel")
    _detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    _detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def restore_frame(task):
    global _restorer
    import cv2
    import torch
    source, dest = task
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"Cannot read {source}")
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
    _detector.setInput(blob)
    detections = _detector.forward()
    if not (detections[0, 0, :, 2] >= 0.7).any():
        shutil.copy2(source, dest)
        return f"No face — skipped {os.path.basename(source)}"
    if _restorer is None:
        from gfpgan import GFPGANer
        _restorer = GFPGANer(model_path=_weights + "/GFPGANv1.4.pth", upscale=1,
                            arch="clean", channel_multiplier=2, bg_upsampler=None,
                            device=torch.device("cpu"))
        # Avoid facexlib downloading weights into its package directory.
        # The download script installs its expected local detection/parsing files.
    _, _, restored = _restorer.enhance(image, has_aligned=False, only_center_face=False,
                                      paste_back=True, weight=0.5)
    if restored is None or not cv2.imwrite(dest, restored):
        raise RuntimeError(f"Face restoration failed for {source}")
    return f"Restored {os.path.basename(source)}"


def restore(source, output, settings, options, ctx):
    output.mkdir()
    inputs = frames(source)
    parallel_frames(restore_frame, init_faces,
                    (str(settings.weights), max(1, min(8, cpu_count() // options.workers))),
                    ((str(p), str(output / p.name)) for p in inputs),
                    options.workers, ctx, 0.5, 0.15)
    verify_frames(output, len(inputs))