import sys
from pathlib import Path
if __name__ == "__main__":
 if sys.path[0] == "" or Path(sys.path[0]) == Path.cwd():
  del sys.path[0]
 from ipykernel import kernelapp as app
 app.launch_new_instance()
 