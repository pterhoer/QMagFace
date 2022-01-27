import os


def convert_backslashes(string: str):
    return string.replace("\\", "/")


def path_join(*paths):
    return convert_backslashes(os.path.join(*paths))


def list_all_files(path):
    paths = []
    for root, _, files in os.walk(path):
        if len(files) < 1:
            continue
        paths += [path_join(root, file) for file in files]
    return paths


def list_all_files_from_path(path):
    if path == '':
        return list_all_files(path)
    files = []
    for file in list_all_files(path):
        files.append(file[len(path) + 1:])
    return files
