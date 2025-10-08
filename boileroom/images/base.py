from modal import Image

base_image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("biotite>=1.0.1")
    )