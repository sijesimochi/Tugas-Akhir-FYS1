import csv
import os


def update_csv_columns(base_dir, sub_dirs, column_names):
    for sub_dir in sub_dirs:
        abs_path = os.path.join(base_dir, sub_dir)

        for (
            root,
            dirs,
            files,
        ) in os.walk(abs_path):
            for file in files:
                if file.endswith("csv"):
                    csv_path = os.path.join(root, file)
                    with open(csv_path, "r") as csv_file:
                        reader = csv.reader(csv_file)
                        rows = list(reader)

                        rows.insert(0, column_names)

                    with open(csv_path, "w") as new_file:
                        writer = csv.writer(new_file)
                        writer.writerows(rows)


sub_dirs = ["bungkuk", "duduk", "jatoh", "jongkok"]

column_names = [
    "timestamp",
    "xpos",
    "ypos",
    "zpos",
    "xvel",
    "yvel",
    "zvel",
    "xacc",
    "yacc",
    "zacc",
]

update_csv_columns(base_dir="C:/CJ/Coolyeah/CD/Coding/Tugas-Akhir-FYS1/dataset", sub_dirs=sub_dirs, column_names=column_names)