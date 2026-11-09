import io
import sys
import contextlib
import traceback
from pathlib import Path
import nbformat as nbf

notebook_path = Path("work/notebooks/capstone.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

print(f"Executing {len(nb.cells)} cells in {notebook_path}...")

# Create execution environment
exec_globals = {"__name__": "__main__"}

for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"--> Running cell {i+1}...")
        cell.outputs = []
        stdout_trap = io.StringIO()
        stderr_trap = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_trap), contextlib.redirect_stderr(stderr_trap):
                exec(cell.source, exec_globals)
            out_str = stdout_trap.getvalue()
            err_str = stderr_trap.getvalue()
            
            if out_str:
                cell.outputs.append(nbf.v4.new_output(output_type="stream", name="stdout", text=out_str))
                print(f"    Output: {len(out_str)} chars")
            if err_str:
                cell.outputs.append(nbf.v4.new_output(output_type="stream", name="stderr", text=err_str))
                print(f"    Stderr: {len(err_str)} chars")
            cell.execution_count = i + 1
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"Error in cell {i+1}:\n{err_msg}")
            cell.outputs.append(nbf.v4.new_output(
                output_type="error",
                ename=type(e).__name__,
                evalue=str(e),
                traceback=err_msg.splitlines()
            ))
            raise

# Save executed notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# Also save mirror to root capstone.ipynb
root_path = Path("capstone.ipynb")
with open(root_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved: {notebook_path} and {root_path}")
