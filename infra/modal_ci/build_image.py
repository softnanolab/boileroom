"""Build the sandbox image ahead of the first job: `python -m infra.modal_ci.build_image`.

Modal caches images by definition, so the controller's `Sandbox.create` reuses this build and a job
does not wait for it. Re-run after changing `image.py` or anything it embeds.
"""

import modal

from infra.modal_ci.controller import RUNNER_APP, sandbox_image

if __name__ == "__main__":
    app = modal.App.lookup(RUNNER_APP, create_if_missing=True)
    with modal.enable_output():
        built = sandbox_image.build(app)
    print(f"sandbox image: {built.object_id}")
