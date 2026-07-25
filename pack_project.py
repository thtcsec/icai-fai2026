"""Script to package the complete ICAI-FAI 2026 project & research artifact into a clean standalone zip file."""
import os
import zipfile

def package_project(output_zip: str = "icai-fai2026-bundle.zip"):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(base_dir, output_zip)

    # Exclude patterns
    exclude_dirs = {".git", "__pycache__", ".venv", "venv", "build", "dist"}
    exclude_extensions = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz", ".pyc"}
    exclude_files = {output_zip, "pack_project.py"}

    print(f"📦 Packaging project files into: {output_zip}...")
    file_count = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_extensions:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)

                zipf.write(full_path, arcname=rel_path)
                file_count += 1

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully created '{output_zip}' with {file_count} files ({file_size_mb:.2f} MB).")

if __name__ == "__main__":
    package_project()
