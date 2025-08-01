import os
import zipfile
import rarfile
import tarfile
import py7zr
import json
import sys
import tempfile
import datetime # Added for timestamp

def is_text_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # Common image magic numbers
            if header.startswith(b'\x89\x50\x4e\x47') or \
               header.startswith(b'\xff\xd8\xff\xe0') or \
               header.startswith(b'\xff\xd8\xff\xe1') or \
               header.startswith(b'GIF8'):
                return False
            f.seek(0)
            chunk = f.read(1024)
            try:
                chunk.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    chunk.decode('latin-1')
                except UnicodeDecodeError:
                    return False
            return True
    except Exception:
        return False

def convert_file_to_markdown(file_path, file_name, level):
    if is_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                if file_name.endswith('.json'):
                    try:
                        content = json.dumps(json.loads(content), indent=4)
                    except json.JSONDecodeError:
                        pass
                return '  ' * level + '# ' + file_name + '\n' + \
                       '  ' * level + '\n' + \
                       '```\n' + content.strip() + '\n' + \
                       '  ' * level + '```\n\n'
        except Exception as e:
            print(f"Warning: Could not read file {file_path} as text: {e}")
            return '  ' * level + '# ' + file_name + ' (Error reading content)\n\n'
    else:
        return '  ' * level + '# ' + file_name + ' (Binary or non-text file)\n\n'

def convert_dir_to_markdown(
    dir_path,
    dir_name,
    level,
    script_dir_to_ignore=None,
    general_exclude_dirs=None,
    general_exclude_files=None,
    is_archive_extraction_context=False,
    archive_extraction_temp_root=None
):
    if general_exclude_dirs is None:
        general_exclude_dirs = []
    if general_exclude_files is None:
        general_exclude_files = []

    markdown_content = '  ' * level + '# ' + dir_name + '\n'

    try:
        entries = sorted(os.listdir(dir_path), key=lambda x: (not os.path.isdir(os.path.join(dir_path, x)), x.lower()))
    except PermissionError:
        print(f"Warning: Permission denied for directory {dir_path}. Skipping.")
        return '  ' * level + f'# {dir_name} (Permission Denied)\n\n'

    norm_archive_extraction_temp_root = None
    if is_archive_extraction_context and archive_extraction_temp_root:
        norm_archive_extraction_temp_root = os.path.normcase(os.path.abspath(archive_extraction_temp_root))


    for item_name in entries:
        item_path = os.path.join(dir_path, item_name)
        normalized_abs_item_path = os.path.normcase(os.path.abspath(item_path))

        if os.path.isdir(item_path):
            if script_dir_to_ignore:
                try:
                    if os.path.samefile(item_path, script_dir_to_ignore):
                        print(f"Ignoring script's directory: {item_path}")
                        continue
                except FileNotFoundError:
                    pass
            
            if item_name.lower() in general_exclude_dirs:
                print(f"Ignoring excluded directory: {item_path}")
                continue
            
            markdown_content += convert_dir_to_markdown(
                item_path, item_name, level + 1,
                script_dir_to_ignore, general_exclude_dirs, general_exclude_files,
                is_archive_extraction_context, archive_extraction_temp_root # Pass context for subdirs
            )
        
        elif os.path.isfile(item_path):
            skip_file = False
            if is_archive_extraction_context and norm_archive_extraction_temp_root:
                # item_path is like C:\temp\extract\foo\bar.txt
                # norm_archive_extraction_temp_root is c:\temp\extract (normcased)
                relative_path_in_archive = os.path.normcase(os.path.relpath(normalized_abs_item_path, norm_archive_extraction_temp_root))
                
                for excluded_abs_path in general_exclude_files: # these are already normcased absolute
                    # Check if excluded_abs_path ends with the relative_path_in_archive, prefixed by a separator
                    # This handles cases like excluding "...\foo\bar.txt" when archive has "foo\bar.txt"
                    expected_suffix = os.sep + relative_path_in_archive
                    if excluded_abs_path.endswith(expected_suffix):
                        skip_file = True
                        break
            else: # Direct filesystem processing
                if normalized_abs_item_path in general_exclude_files: # general_exclude_files are normcased absolute
                    skip_file = True
            
            if skip_file:
                print(f"Ignoring excluded file: {item_path}")
                continue
            
            markdown_content += convert_file_to_markdown(item_path, item_name, level + 1)
            
    return markdown_content

def extract_archive(archive_path, script_dir_to_ignore=None, general_exclude_dirs=None, general_exclude_files=None):
    if general_exclude_dirs is None:
        general_exclude_dirs = []
    if general_exclude_files is None:
        general_exclude_files = []
        
    markdown_content = '# Archive: ' + os.path.basename(archive_path) + '\n\n'

    with tempfile.TemporaryDirectory(prefix="archive_extract_") as temp_dir:
        print(f"Extracting {archive_path} to temporary directory {temp_dir}...")
        extracted_something = False
        norm_temp_dir = os.path.normcase(os.path.abspath(temp_dir))
        try:
            # ... (extraction logic for zip, rar, tar, 7z - no changes here) ...
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    extracted_something = True
            elif archive_path.endswith('.rar'):
                try:
                    with rarfile.RarFile(archive_path, 'r') as rar_ref:
                        rar_ref.extractall(temp_dir)
                        extracted_something = True
                except rarfile.NeedFirstVolume:
                    print(f"Error: Multi-volume RAR archives are not fully supported or first volume is missing for {archive_path}.")
                    markdown_content += f"Error: Multi-volume RAR archive {os.path.basename(archive_path)} - processing may be incomplete.\n"
                except Exception as e:
                    print(f"Error extracting RAR {archive_path}: {e}")
                    markdown_content += f"Error extracting RAR {os.path.basename(archive_path)}: {e}\n"

            elif archive_path.endswith(('.tar', '.tar.gz', '.tar.bz2', '.tgz', '.tbz2')):
                mode = 'r:*'
                if archive_path.endswith('.tar'): mode = 'r:'
                elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'): mode = 'r:gz'
                elif archive_path.endswith('.tar.bz2') or archive_path.endswith('.tbz2'): mode = 'r:bz2'

                with tarfile.open(archive_path, mode) as tar_ref:
                    # To prevent TarSlip vulnerability, filter members if Python >= 3.12 or use a loop with check
                    if sys.version_info >= (3, 12):
                         tar_ref.extractall(temp_dir, filter='data')
                    else: # Manual check for older Python versions
                        for member in tar_ref.getmembers():
                            member_path = os.path.join(temp_dir, member.name)
                            # Basic check to prevent extracting outside temp_dir
                            if not os.path.abspath(member_path).startswith(os.path.abspath(temp_dir)):
                                print(f"Warning: Skipping potentially unsafe tar member: {member.name}")
                                continue
                            tar_ref.extract(member, path=temp_dir)
                    extracted_something = True
            elif archive_path.endswith('.7z'):
                with py7zr.SevenZipFile(archive_path, 'r') as sz_ref:
                    sz_ref.extractall(temp_dir)
                    extracted_something = True
            else:
                raise ValueError('Unsupported archive format')


            if extracted_something:
                entries = sorted(os.listdir(temp_dir), key=lambda x: (not os.path.isdir(os.path.join(temp_dir, x)), x.lower()))
                for item_name in entries: # These are items at the root of the archive
                    item_path = os.path.join(temp_dir, item_name)
                    normalized_abs_item_path = os.path.normcase(os.path.abspath(item_path))
                    
                    if os.path.isdir(item_path):
                        if script_dir_to_ignore:
                            try: # Heuristic: compare basenames if found in archive root
                                if os.path.basename(item_path).lower() == os.path.basename(script_dir_to_ignore).lower():
                                     print(f"Ignoring script's directory (found in archive root): {item_path}")
                                     continue
                            except FileNotFoundError:
                                pass
                        
                        if item_name.lower() in general_exclude_dirs:
                            print(f"Ignoring excluded directory from archive: {item_path}")
                            continue
                        
                        markdown_content += convert_dir_to_markdown(
                            item_path, item_name, 1, 
                            script_dir_to_ignore, general_exclude_dirs, general_exclude_files,
                            is_archive_extraction_context=True, archive_extraction_temp_root=temp_dir
                        )
                    
                    elif os.path.isfile(item_path):
                        skip_file = False
                        # item_name is already the relative path for files at archive root
                        relative_path_in_archive = os.path.normcase(item_name.replace('/', os.sep)) 
                        
                        for excluded_abs_path in general_exclude_files:
                            expected_suffix = os.sep + relative_path_in_archive
                            if excluded_abs_path.endswith(expected_suffix):
                                skip_file = True
                                break
                        
                        if skip_file:
                            print(f"Ignoring excluded file from archive root: {item_name}")
                            continue
                        markdown_content += convert_file_to_markdown(item_path, item_name, 1)
                        
        except FileNotFoundError:
             raise ValueError(f"Archive file not found: {archive_path}")
        except (zipfile.BadZipFile, tarfile.ReadError, rarfile.BadRarFile, py7zr.exceptions.ArchiveError if hasattr(py7zr, 'exceptions') and hasattr(py7zr.exceptions, 'ArchiveError') else Exception) as e:
            if "py7zr" in str(type(e)).lower() and not (hasattr(py7zr, 'exceptions') and hasattr(py7zr.exceptions, 'ArchiveError') and isinstance(e, py7zr.exceptions.ArchiveError)):
                 pass
            raise ValueError(f"Could not read archive {archive_path}. It might be corrupted or an unsupported subtype. Error: {e}")
        except Exception as e:
            raise ValueError(f"An unexpected error occurred during extraction of {archive_path}: {e}")

    return markdown_content

def load_config(config_path):
    config = {}
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                print(f"Warning: Malformed line {line_number} in '{config_path}': '{line}'. Skipping.")
                continue
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    return config

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, 'config.txt')
    config = load_config(config_file_path)

    target_path_from_config = config.get('object')
    base_output_filename = config.get('output_filename', 'output')
    
    exclude_dirs_str = config.get('exclude_dirs', '')
    general_exclude_dirs = [d.strip().lower() for d in exclude_dirs_str.split(',') if d.strip()]
    if general_exclude_dirs:
        print(f"Configured to exclude directory names: {', '.join(general_exclude_dirs)}")

    exclude_files_str = config.get('exclude_files', '')
    raw_exclude_files_paths = [p.strip() for p in exclude_files_str.split(',') if p.strip()]
    # Normalize paths from config for consistent comparison
    general_exclude_files = [os.path.normcase(os.path.abspath(p)) for p in raw_exclude_files_paths]
    if general_exclude_files:
        print(f"Configured to exclude specific files: {', '.join(raw_exclude_files_paths)}")


    if not target_path_from_config:
        print(f"Error: 'object' not defined in '{config_file_path}'. Add 'object=/path/to/target'.")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base, output_filename_ext = os.path.splitext(base_output_filename)
    if not output_filename_ext: output_filename_ext = ".md"
    if output_filename_base.lower().endswith('.md') and output_filename_ext.lower() == '.md':
        output_filename_base = output_filename_base[:-3]
    output_filename_with_ts = f"{output_filename_base}_{timestamp}{output_filename_ext}"
    output_file_path = os.path.join(script_dir, output_filename_with_ts)

    target_path = os.path.abspath(os.path.join(script_dir, target_path_from_config) if not os.path.isabs(target_path_from_config) else target_path_from_config)

    if not os.path.exists(target_path):
        print(f"Error: Target path '{target_path}' (from config: '{target_path_from_config}') does not exist.")
        sys.exit(1)

    markdown_content = ""
    try:
        if os.path.isdir(target_path):
            print(f"Processing directory: {target_path}")
            markdown_content = convert_dir_to_markdown(
                target_path, os.path.basename(target_path) or "Root Directory", 0,
                script_dir_to_ignore=script_dir,
                general_exclude_dirs=general_exclude_dirs,
                general_exclude_files=general_exclude_files,
                is_archive_extraction_context=False, # Explicitly False for direct directory
                archive_extraction_temp_root=None
            )
        elif os.path.isfile(target_path):
            is_known_archive = any(target_path.lower().endswith(ext) for ext in
                                   ['.zip', '.rar', '.tar', '.tar.gz', '.tar.bz2', '.tgz', '.tbz2', '.7z'])
            if is_known_archive:
                print(f"Processing archive: {target_path}")
                markdown_content = extract_archive(
                    target_path, 
                    script_dir_to_ignore=script_dir,
                    general_exclude_dirs=general_exclude_dirs,
                    general_exclude_files=general_exclude_files
                )
            elif is_text_file(target_path):
                print(f"Processing single text file: {target_path}")
                # Check if this single text file should be excluded
                normalized_target_path = os.path.normcase(os.path.abspath(target_path))
                if normalized_target_path in general_exclude_files:
                    print(f"Ignoring excluded single file: {target_path}")
                    markdown_content = f"# {os.path.basename(target_path)} (File excluded by configuration)\n"
                else:
                    markdown_content = convert_file_to_markdown(target_path, os.path.basename(target_path), 0)
            else:
                print(f"Target file '{target_path}' is not a recognized archive and not identified as a text file. Skipping.")
                markdown_content = f"# {os.path.basename(target_path)} (Non-text file, not an archive)\n"
        else:
            print(f"Error: Target path '{target_path}' is not a file or directory.")
            sys.exit(1)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f'Conversion complete. Output file: {output_file_path}')

    except ValueError as ve:
        print(f'Error: {str(ve)}')
    except Exception as e:
        print(f'An unexpected error occurred: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if hasattr(sys, '_MEIPASS'):
        os.environ['P7ZIP_PATH'] = sys._MEIPASS

    if hasattr(py7zr, 'discover_library') and \
       hasattr(py7zr, 'exceptions') and \
       hasattr(py7zr.exceptions, 'LibraryError'):
        try:
            py7zr.discover_library()
        except py7zr.exceptions.LibraryError as e:
            print(f"Warning for 7z support: {e}")
            print("Ensure 7-Zip (Windows) or p7zip (Linux/macOS) is installed/accessible,")
            print("or 7z library (7z.dll/so) is in system's library path or P7ZIP_PATH.")
        except Exception as e_other:
             print(f"Unexpected error during py7zr library discovery: {e_other}")
    else:
        print("Note: Installed py7zr version may be older. Advanced library discovery check skipped.")
        print(f"py7zr version: {getattr(py7zr, '__version__', 'N/A')}")

    main()