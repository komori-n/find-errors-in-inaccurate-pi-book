import csv
import os
import sys

ANS_FILENAME = "pi-1m.txt"


def print_all_mistakes(s_ans: str, s_test: str, idx_offset: int):
    assert len(s_ans) == len(s_test)

    skip_until = 0
    for i in range(len(s_ans)):
        if i < skip_until:
            continue

        if s_ans[i] != s_test[i]:
            chunk_start = i // 10 * 10
            chunk_end = chunk_start + 10
            substr_ans = s_ans[chunk_start:chunk_end]
            substr_test = s_test[chunk_start:chunk_end]
            skip_until = chunk_end

            idx = idx_offset + chunk_start
            row = idx // 100
            col = (idx // 10) % 10
            print(
                f"{str(row).zfill(5)}, {str(col).zfill(5)}",
                f"ans:{substr_ans} book:{substr_test}",
            )


def main(datapath: str):
    filepath = os.path.join(os.path.dirname(__file__), ANS_FILENAME)

    ans = ""
    with open(filepath, "r") as f:
        ans = "".join(f.readlines()).strip()

    ans = ans[2:]  # skip "3."

    for i in range(1, 4):
        csvpath = os.path.join(datapath, str(i).zfill(3) + ".csv")

        x = ""
        with open(csvpath, "r") as f:
            reader = csv.reader(f)
            x = "".join(["".join(r) for r in reader])

        print_all_mistakes(ans[(i - 1) * 10000 : i * 10000], x, (i - 1) * 10000)


if __name__ == "__main__":
    argv = sys.argv
    data_path = "./data"
    if len(argv) > 1:
        data_path = argv[1]
        if not os.path.isdir(data_path):
            print(f"usage {argv[0]} datadir")

    main(data_path)
