import os
import sys
from urllib.parse import quote

print(sys.executable)
print(os.path.abspath("."))

INDENT = "    "
MAX_LINE_WIDTH = 300
ROOT = "https://github.com/pengzhenghao/test-mmlab-reading-groups/tree/master/"


def get_url(string, top_url=None):
    return os.path.join(top_url or ROOT, quote(string))


def listdir(path):
    ret = dict()
    for p in os.listdir(path):
        if p.startswith("."):
            continue
        abs_path = os.path.join(path, p)
        if os.path.isdir(abs_path):
            print(abs_path)
            ret[p] = listdir(abs_path)
        else:
            continue
    return ret


def _parse_folders(folders, url):
    lines = []

    folders = {k: folders[k] for k in sorted(folders.keys())}
    if not folders:
        return []

    for k, v in folders.items():
        current_url = get_url(k, url)
        l = "* [{}]({})".format(k, current_url)

        sub_folder_lines = _parse_folders(v, current_url)
        sub_folder_lines = [INDENT + l for l in sub_folder_lines]

        lines.append(l)
        lines.extend(sub_folder_lines)

    return lines


def parse_folders(folders):
    new_lines = _parse_folders(folders, url=None)
    new_lines = [l[:MAX_LINE_WIDTH] + "\n" for l in new_lines]
    return new_lines


def parse_existing(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    existing_lines = []
    detected = False
    for line in lines:
        existing_lines.append(line)
        if "DIVIDER_DO_NOT_EDIT_ANYTHING_BELOW_THIS_LINE" in line:
            detected = True
            break
    assert detected, "The divider line is deleted!"
    return existing_lines


def main():
    # Scan all subfolders
    folders = listdir(".")

    # Collect all informations
    new_lines = parse_folders(folders)

    # Read the headers of the file
    file_name = "README.md"
    existing_lines = parse_existing(file_name)

    # Write lines to file
    write_lines = existing_lines + new_lines
    with open(file_name, "w") as f:
        f.writelines(write_lines)

    print("Finished writing files! Previous number of lines {}, now {}.".format(
        len(lines), len(write_lines)
    ))


if __name__ == '__main__':
    # main()
    print("===")
    ret = listdir(".")
    print("===")
    lines = parse_folders(ret)
    for l in lines:
        print(l)
    print("===")
    main()
